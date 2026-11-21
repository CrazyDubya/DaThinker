"""Actual CLI/orchestrator request construction with inert canned transport only."""
import io
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.environ.get('DATHINKER_SOURCE', str(Path(__file__).resolve().parents[1] / 'src')))
from rich.console import Console
from dathinker import cli
from dathinker.orchestrator import ThinkingOrchestrator

CONSTRAINT = 'FIXTURE_CONSTRAINT_LOCAL'
GOAL = 'FIXTURE_GOAL_TIME'
OPEN = 'FIXTURE_ASSUMPTION_OPEN'
CLOSED = 'FIXTURE_ASSUMPTION_CLOSED'
CALLER = 'FIXTURE_CALLER_CONTEXT'
MARKERS = [CONSTRAINT, GOAL, OPEN]


class CannedTransport:
    def __init__(self):
        self.requests = []

    async def chat(self, messages, **kwargs):
        self.requests.append([(m.role, m.content) for m in messages])
        return 'What evidence would help compare the choices?'


def body(request):
    return '\n'.join(content for role, content in request)


def controls(orch):
    orch.set_goal(GOAL)
    orch.add_constraint(CONSTRAINT)
    orch.pin_assumption(OPEN)
    closed = orch.pin_assumption(CLOSED)
    closed.status = 'confirmed'


class ModeContextRequests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dathinker-mode-context-')
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {'HOME': self.temp.name, 'XDG_CONFIG_HOME': self.temp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.network = []
        def blocked(*args, **kwargs):
            self.network.append(1)
            raise AssertionError('Network prohibited')
        for target in ['socket.socket.connect', 'socket.create_connection']:
            guard = patch(target, blocked)
            guard.start()
            self.addCleanup(guard.stop)

    async def asyncTearDown(self):
        self.assertEqual(self.network, [])

    async def test_actual_cli_single_parallel_adaptive_and_reset(self):
        transport = CannedTransport()
        inputs = iter([
            'Synthetic session', '/constraint ' + CONSTRAINT, '/goal ' + GOAL,
            '/pin ' + OPEN, '/agent socratic', 'First fixture turn.',
            '/mode parallel', 'Second fixture turn.', '/mode adaptive',
            'Third fixture turn.', '/reset', 'Fresh synthetic session',
            'Fresh fixture turn.', '/quit'
        ])
        output = io.StringIO()
        with patch.object(cli, 'OpenRouterClient', return_value=transport), \
             patch.object(cli.Prompt, 'ask', side_effect=lambda *a, **k: next(inputs)), \
             patch.object(cli, 'console', Console(file=output, force_terminal=False)):
            await cli.run_interactive_session()
        self.assertNotIn('Error:', output.getvalue())
        for acknowledgement in ['Added constraint', 'Goal set', 'Pinned assumption', 'Started new session']:
            self.assertIn(acknowledgement, output.getvalue())
        self.assertGreaterEqual(len(transport.requests), 8)
        controlled = [r for r in transport.requests if r[-1][1] != 'Fresh fixture turn.']
        fresh = [r for r in transport.requests if r[-1][1] == 'Fresh fixture turn.']
        self.assertTrue(controlled and fresh)
        for request in controlled:
            for marker in MARKERS:
                self.assertEqual(body(request).count(marker), 1)
            current = request[-1][1]
            self.assertEqual(sum(role == 'user' and content == current for role, content in request), 1)
        for request in fresh:
            for marker in MARKERS:
                self.assertNotIn(marker, body(request))
            self.assertNotIn('First fixture turn.', body(request))

    async def test_single_and_parallel_preserve_caller_context_and_open_only_selection(self):
        for mode in ['single', 'parallel']:
            with self.subTest(mode=mode):
                transport = CannedTransport()
                orch = ThinkingOrchestrator(transport)
                orch.start_session('Synthetic')
                controls(orch)
                if mode == 'single':
                    await orch.think_with_agent('Unique fixture turn.', 'socratic', context=CALLER)
                else:
                    await orch.think_parallel('Unique fixture turn.', context=CALLER)
                for request in transport.requests:
                    for marker in MARKERS + [CALLER]: self.assertEqual(body(request).count(marker), 1)
                    self.assertNotIn(CLOSED, body(request))
                    self.assertEqual(sum(role == 'user' and text == 'Unique fixture turn.' for role, text in request), 1)
                self.assertEqual(sum(h['type'] == 'user' for h in orch.active_session.history), 1)
                self.assertTrue(all(len(a.conversation_history) in [0, 2] for a in orch.agents.values()))

    async def test_adaptive_existing_context_format_retained(self):
        transport = CannedTransport()
        orch = ThinkingOrchestrator(transport)
        orch.start_session('Synthetic')
        controls(orch)
        await orch.think_adaptive('Unique fixture turn.', context=CALLER)
        expected = f'WORKING ASSUMPTIONS:\n- {OPEN}\nCONSTRAINTS:\n- {CONSTRAINT}\nGOAL: {GOAL}\n{CALLER}'
        for request in transport.requests:
            self.assertIn(('system', 'Session Context:\n' + expected), request)
            self.assertNotIn(CLOSED, body(request))
        self.assertEqual(sum(h['type'] == 'user' for h in orch.active_session.history), 1)

    async def test_no_session_or_controls_and_new_session_do_not_leak(self):
        transport = CannedTransport()
        orch = ThinkingOrchestrator(transport)
        await orch.think_with_agent('No session fixture.', 'socratic', context=CALLER)
        self.assertIn(('system', 'Session Context:\n' + CALLER), transport.requests[-1])
        orch.start_session('Controlled')
        controls(orch)
        await orch.think_parallel('Old fixture turn.')
        orch.start_session('Fresh')
        transport.requests.clear()
        await orch.think_parallel('New fixture turn.')
        for request in transport.requests:
            for marker in MARKERS: self.assertNotIn(marker, body(request))
            self.assertNotIn('Old fixture turn.', body(request))
            self.assertFalse(any(role == 'system' and text.startswith('Session Context:') for role, text in request))
        self.assertEqual(sum(h['type'] == 'user' for h in orch.active_session.history), 1)

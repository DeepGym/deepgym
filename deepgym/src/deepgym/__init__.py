"""DeepGym — Managed RL training and evaluation infrastructure."""

__version__ = '0.6.1'

from deepgym.async_core import AsyncDeepGym
from deepgym.benchmark_envs import (
    MixedEnvironment,
    PatchVerifier,
    SWEBenchProEnvironment,
    TerminalBenchEnvironment,
)
from deepgym.benchmark_ops import BenchmarkAuditReport, BenchmarkLeak, BenchmarkManifestEntry
from deepgym.computer_use import ComputerUseEnvironment
from deepgym.core import DeepGym
from deepgym.gym import AsyncDeepGymEnv, DeepGymEnv, GymInfo, GymObservation
from deepgym.integrations.reward import AsyncRewardFunction, RewardFunction
from deepgym.models import (
    Action,
    BatchResult,
    CaseResult,
    Environment,
    EvalResult,
    MultiTurnEnvironment,
    Observation,
    RunResult,
    Trajectory,
    VerifierResult,
)
from deepgym.registry import list_environments, load_environment, load_suite
from deepgym.reward_qa import RewardAuditor, VerifierAuditReport
from deepgym.session import Session, StepResult
from deepgym.verifier import Verifier

__all__ = [
    '__version__',
    'Action',
    'AsyncDeepGym',
    'AsyncDeepGymEnv',
    'AsyncRewardFunction',
    'BatchResult',
    'BenchmarkAuditReport',
    'BenchmarkLeak',
    'BenchmarkManifestEntry',
    'ComputerUseEnvironment',
    'DeepGym',
    'DeepGymEnv',
    'Environment',
    'EvalResult',
    'GymInfo',
    'GymObservation',
    'MixedEnvironment',
    'MultiTurnEnvironment',
    'Observation',
    'PatchVerifier',
    'RewardAuditor',
    'RewardFunction',
    'RunResult',
    'Session',
    'StepResult',
    'SWEBenchProEnvironment',
    'TerminalBenchEnvironment',
    'CaseResult',
    'Trajectory',
    'Verifier',
    'VerifierAuditReport',
    'VerifierResult',
    'list_environments',
    'load_environment',
    'load_suite',
]

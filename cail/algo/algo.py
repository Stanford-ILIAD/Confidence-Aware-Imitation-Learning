from .ppo import PPO, PPOExpert
from .sac import SAC, SACExpert
from .gail import GAIL, GAILExpert
from .airl import AIRL, AIRLExpert
from .cail import CAIL, CAILExpert
from .twoiwil import TwoIWIL, TwoIWILExpert
from .icgail import ICGAIL, ICGAILExpert
from .trex import TREX, TREXExpert
from .drex import DREX, DREXExpert, DREXBCExpert
from .ssrr import SSRR, SSRRExpert


# all the algorithms
ALGOS = {
    'sac': SAC,
    'ppo': PPO,
    'gail': GAIL,
    'airl': AIRL,
    'cail': CAIL,
    '2iwil': TwoIWIL,
    'icgail': ICGAIL,
    'trex': TREX,
    'drex': DREX,
    'ssrr': SSRR,
}

# all the well-trained algorithms
EXP_ALGOS = {
    'sac': SACExpert,
    'ppo': PPOExpert,
    'gail': GAILExpert,
    'airl': AIRLExpert,
    'cail': CAILExpert,
    '2iwil': TwoIWILExpert,
    'icgail': ICGAILExpert,
    'trex': TREXExpert,
    'drex': DREXExpert,
    'drex_bc': DREXBCExpert,
    'ssrr': SSRRExpert,
}
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import (
    SACAgentHybridSingleArm,
)
from serl_launcher.agents.continuous.sac_hybrid_dual import (
    SACAgentHybridDualArm,
)

agents = {
    "bc": BCAgent,
    "sac": SACAgent,
    "sac_hybrid_single": SACAgentHybridSingleArm,
    "sac_hybrid_dual": SACAgentHybridDualArm,
}

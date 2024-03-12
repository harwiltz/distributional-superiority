OPTIONS := "lim_malek"

agent := "dsup"
seed := "0"
num_steps := "200000"
option_idx := "0"
env := OPTIONS

DQN   := "config.agents.dqn"
DAU   := "config.agents.dau"
QRDQN := "config.agents.qrdqn"
DSUP  := "config.agents.dsup"

agent_config := "config.agents." + agent

EVAL_EPISODES_OPTIONS := "200"

writer := "aim"

time_mul := "1"
risk_type := "neutral"
risk_param := "1.0"

WORKDIR := "models" / env / agent / "h-" + time_mul + "_risk-" + risk_type + "-" + risk_param + "_v" + seed

train agent env eval_episodes:
    python -m dsup.testbed \
        --agent config:{{agent_config}}.{{env}} \
        --agent set:seed={{seed}} \
        --agent set:num_steps={{num_steps}} \
        --agent set:warmup_steps=0 \
        --agent fiddler:config.fiddlers.set_decision_frequency\(decision_frequency={{time_mul}}\) \
        --experiment {{env}} \
        --metric_writer {{writer}} \
        --workdir {{WORKDIR}} \
        --num_eval_episodes {{eval_episodes}}

train_dsup_shifted agent env eval_episodes:
    python -m dsup.testbed \
        --agent config:{{agent_config}}.{{env}} \
        --agent set:seed={{seed}} \
        --agent set:num_steps={{num_steps}} \
        --agent set:warmup_steps=0 \
        --agent fiddler:config.fiddlers.set_decision_frequency\(decision_frequency={{time_mul}}\) \
        --agent fiddler:config.fiddlers.dsup_shifted\(rescale_factor=0.5\) \
        --experiment {{env}} \
        --metric_writer {{writer}} \
        --workdir {{WORKDIR}} \
        --enable_checkpoints \
        --num_eval_episodes {{eval_episodes}}
        
OPTION_IDX := OPTIONS + "\\(idx=" + option_idx + "\\)"
        
train_options: (train agent OPTION_IDX EVAL_EPISODES_OPTIONS)
train_options_dsup_shifted: (train_dsup_shifted agent OPTION_IDX EVAL_EPISODES_OPTIONS)

train_risky agent env eval_episodes:
    python -m dsup.testbed \
        --agent config:{{agent_config}}.{{env}} \
        --agent set:seed={{seed}} \
        --agent set:num_steps={{num_steps}} \
        --agent fiddler:config.fiddlers.set_decision_frequency\(decision_frequency={{time_mul}}\) \
        --agent fiddler:config.fiddlers.set_risk_measure_cvar\({{risk_param}}\) \
        --experiment {{env}}-risky \
        --metric_writer {{writer}} \
        --workdir {{WORKDIR}} \
        --enable_checkpoints \
        --num_eval_episodes {{eval_episodes}}

train_options_risky: (train_risky agent OPTION_IDX EVAL_EPISODES_OPTIONS)
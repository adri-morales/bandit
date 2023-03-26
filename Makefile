# Define variables
DATA_DIR := data
PROCESSED_DIR := $(DATA_DIR)/processed
RESULTS_DIR := $(DATA_DIR)/results

PYTHON := python
TRAIN_SCRIPT := src/train.py
AGENT_DIR := src/agents

N_ACTIONS := 10
LR := 0.1
EPSILON := 0.1
QI := 3
C := 2

# Define targets and dependencies
$(RESULTS_DIR)/greedy_history.csv: $(AGENT_DIR)/greedy.py $(TRAIN_SCRIPT) | $(PROCESSED_DIR) $(RESULTS_DIR)
	$(PYTHON) $(TRAIN_SCRIPT) -a greedy -p n_actions:$(N_ACTIONS) lr:$(LR)

$(RESULTS_DIR)/epsilon_greedy_history.csv: $(AGENT_DIR)/egreedy.py $(TRAIN_SCRIPT) | $(PROCESSED_DIR) $(RESULTS_DIR)
	$(PYTHON) $(TRAIN_SCRIPT) -a epsilon_greedy -p n_actions:$(N_ACTIONS) lr:$(LR) epsilon:$(EPSILON)

$(RESULTS_DIR)/optimistic_greedy_history.csv: $(AGENT_DIR)/optimistic.py $(TRAIN_SCRIPT) | $(PROCESSED_DIR) $(RESULTS_DIR)
	$(PYTHON) $(TRAIN_SCRIPT) -a optimistic_greedy -p n_actions:$(N_ACTIONS) lr:$(LR) qi:$(QI)

$(RESULTS_DIR)/ucb_history.csv: $(AGENT_DIR)/ucb.py $(TRAIN_SCRIPT) | $(PROCESSED_DIR) $(RESULTS_DIR)
	$(PYTHON) $(TRAIN_SCRIPT) -a ucb -p n_actions:$(N_ACTIONS) lr:$(LR) c:$(C)

$(PROCESSED_DIR):
	mkdir -p $@
$(RESULTS_DIR):
	mkdir -p $@

# Define a rule that depends on all of the individual rules
.PHONY: train_all
train_all: $(RESULTS_DIR)/greedy_history.csv $(RESULTS_DIR)/epsilon_greedy_history.csv $(RESULTS_DIR)/optimistic_greedy_history.csv $(RESULTS_DIR)/ucb_history.csv

.PHONY: clean
clean:
	rm -rf $(RESULTS_DIR) $(PROCESSED_DIR)
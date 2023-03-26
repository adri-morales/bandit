# Define variables
DATA_DIR := data
PROCESSED_DIR := $(DATA_DIR)/processed
RESULTS_DIR := results

PYTHON := python
TRAIN_SCRIPT := src/train.py
AGENT_DIR := src/agents

# Define targets and dependencies
$(RESULTS_DIR)/greedy_history.csv: $(wildcard $(AGENT_DIR)/greedy/*) $(TRAIN_SCRIPT) | $(PROCESSED_DIR) $(RESULTS_DIR)
	$(PYTHON) $(TRAIN_SCRIPT) -c $(AGENT_DIR)/greedy/greedy.yaml

$(RESULTS_DIR)/epsilon_greedy_history.csv: $(wildcard $(AGENT_DIR)/egreedy/*) $(TRAIN_SCRIPT) | $(PROCESSED_DIR) $(RESULTS_DIR)
	$(PYTHON) $(TRAIN_SCRIPT) -c $(AGENT_DIR)/egreedy/egreedy.yaml

$(RESULTS_DIR)/optimistic_greedy_history.csv: $(wildcard $(AGENT_DIR)/optimistic/*) $(TRAIN_SCRIPT) | $(PROCESSED_DIR) $(RESULTS_DIR)
	$(PYTHON) $(TRAIN_SCRIPT) -c $(AGENT_DIR)/optimistic/optimistic.yaml

$(RESULTS_DIR)/ucb_history.csv: $(wildcard $(AGENT_DIR)/ucb/*) $(TRAIN_SCRIPT) | $(PROCESSED_DIR) $(RESULTS_DIR)
	$(PYTHON) $(TRAIN_SCRIPT) -c $(AGENT_DIR)/ucb/ucb.yaml

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
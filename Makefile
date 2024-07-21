RED := \033[31m
GREEN := \033[32m
RESET := \033[0m

all: venv install

venv:
	@echo "$(GREEN)creating virtual environnement...$(RESET)" 
	@python -m venv VirtualEnvironnement
	@echo "$(GREEN)Done.$(RESET)"

install:
	@echo "$(GREEN)installing dependencies..."
	@. VirtualEnvironnement/bin/activate && pip install -r requirements.txt
	@echo "Done.$(RESET)"

clean:
	@echo "$(RED)removing virtual environnement...$(RESET)"
	@rm -rf VirtualEnvironnement
	@echo "$(RED)Done.$(RESET)"

.PHONY: all venv install clean 
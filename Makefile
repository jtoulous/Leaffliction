RED := \033[31m
GREEN := \033[32m
RESET := \033[0m

all: venv install unzip activate

venv:
	@echo "$(GREEN)Creating virtual environnement...$(RESET)"
	@python -m venv venv
	@echo "$(GREEN) --> Done.$(RESET)"

install:
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	@. venv/bin/activate && pip install -r requirements.txt
	@echo "$(GREEN) --> Done.$(RESET)"

unzip:
	@echo "$(GREEN)Unzipping data files...$(RESET)"
	@unzip -o data/leaves.zip -d data/leaves
	@mv data/leaves/images/* data/leaves/
	@rm -rf data/leaves/images/
	@echo "$(GREEN) --> Done.$(RESET)"

activate:
	@echo "$(GREEN) To activate the virtual environment, run: $(RESET)"
	@echo "   '. venv/bin/activate'"
	@echo "\n$(GREEN) To deactivate the virtual environment, run: $(RESET)"
	@echo "   'deactivate'"

clean:
	@echo "$(GREEN)Cleaning...$(RESET)"
	@if [ -d __pycache__ ] || [ -d srcs/__pycache__ ]; then \
		echo "$(GREEN)     --> Removing __pycache__ folders..."; \
		find . -type d -name '__pycache__' -exec rm -rf {} +; \
	fi
	@if [ -d data/leaves ]; then \
		echo "$(GREEN)     --> Removing leaves folder..."; \
		rm -rf data/leaves; \
	fi
	@echo "$(GREEN) --> Done.$(RESET)"; \

fclean: clean
	@echo "$(GREEN)Removing virtual environnement...$(RESET)"
	@rm -rf venv
	@echo "$(GREEN) --> Done.$(RESET)"

re: fclean all

.PHONY: all venv install unzip activate clean fclean re

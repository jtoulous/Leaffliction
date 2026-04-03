RED := \033[31m
GREEN := \033[32m
RESET := \033[0m

all: venv install unzip activate

venv:
	@echo "$(GREEN)Creating virtual environnement...$(RESET)"
	@python -m venv ~/goinfre/venv
	@echo "$(GREEN) --> Done.$(RESET)"

install:
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	@. ~/goinfre/venv/bin/activate && pip install -r requirements.txt
	@echo "$(GREEN) --> Done.$(RESET)"

activate:
	@echo "$(GREEN) To activate the virtual environment, run: $(RESET)"
	@echo "   '. ~/goinfre/venv/bin/activate'"
	@echo "\n$(GREEN) To deactivate the virtual environment, run: $(RESET)"
	@echo "   'deactivate'"
	@echo "\n$(GREEN)Launching the Gradio app...$(RESET)"
	@. ~/goinfre/venv/bin/activate && gradio app.py

clean:
	@echo "$(GREEN)Cleaning...$(RESET)"
	@if [ -d __pycache__ ] || [ -d srcs/__pycache__ ]; then \
		echo "$(GREEN)     --> Removing __pycache__ folders..."; \
		find . -type d -name '__pycache__' -exec rm -rf {} +; \
	fi
	@if [ -d data/leaves ]; then \
		echo "$(GREEN)     --> Removing augmented images from leaves folder..."; \
		find data/leaves -type f -name '*_*' -delete; \
	fi
	@if [ -d data/leaves_preprocessed ]; then \
		echo "$(GREEN)     --> Removing leaves_preprocessed folder..."; \
		rm -rf data/leaves_preprocessed; \
	fi
	@if [ -f data/leaves_preprocessed.zip ]; then \
		echo "$(GREEN)     --> Removing leaves_preprocessed.zip..."; \
		rm -rf data/leaves_preprocessed.zip; \
	fi
	@echo "$(GREEN) --> Done.$(RESET)"; \

fclean: clean
	@echo "$(GREEN)Removing virtual environnement...$(RESET)"
	@rm -rf venv
	@rm -rf ~/goinfre/venv
	@echo "$(GREEN) --> Done.$(RESET)"

re: fclean all

.PHONY: all venv install unzip activate clean fclean re

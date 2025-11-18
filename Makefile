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
	@mv data/leaves/images/* data/leaves/ -n
	@rm -rf data/leaves/images/
	@echo "$(GREEN) --> Done.$(RESET)"

activate:
	@echo "$(GREEN) To activate the virtual environment, run: $(RESET)"
	@echo "   '. venv/bin/activate'"
	@echo "\n$(GREEN) To deactivate the virtual environment, run: $(RESET)"
	@echo "   'deactivate'"

zip:
	@echo "$(GREEN)Zipping data files...$(RESET)"
	@mkdir -p data/leaves_preprocessed/images
	@find data/leaves_preprocessed -mindepth 1 -maxdepth 1 ! -name images -exec cp -r {} data/leaves/images/ \;
	@cd data/leaves_preprocessed && zip -r ../../data/leaves_preprocessed.zip images
	@rm -rf data/leaves/images/
	@echo "$(GREEN) --> Done.$(RESET)"

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
	@echo "$(GREEN) --> Done.$(RESET)"

re: fclean all

.PHONY: all venv install unzip activate clean fclean re

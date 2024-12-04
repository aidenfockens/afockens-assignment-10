# Variables
PYTHON = python
APP = app.py
VENV = venv
REQUIREMENTS = requirements.txt



# Default target
.PHONY: run
run: $(APP)
	$(PYTHON) $(APP)

# Create a virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)

# Install dependencies
.PHONY: install
install: venv
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r $(REQUIREMENTS)

# Clean up environment
.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Run the Flask app using the virtual environment
.PHONY: run-venv
run-venv: $(APP)
	$(VENV)/bin/python $(APP)


#make run: will run the app directly

#make install: will set up the venv and install the dependencies

#make run-venv: run the app in the venv

#make clean: cleans up the venv
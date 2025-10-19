.PHONY: venv graph clean-venv

VENV?=.venv
PYTHON?=python3
CSV?=results.csv
OUTPUT_DIR?=plots
PLOTS?=usl_throughput

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

venv: $(VENV)/bin/activate

graph: $(VENV)/bin/activate
	$(VENV)/bin/python graph.py --csv $(CSV) --output-dir $(OUTPUT_DIR) --plots $(PLOTS)

clean-venv:
	rm -rf $(VENV)

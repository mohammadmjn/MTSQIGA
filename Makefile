SHELL := /bin/bash
.DEFAULT_GOAL := setup

setup:
	pip install -r requirements.txt

serve:
	py -3.6 main.py

lint:
	pylama main.py src

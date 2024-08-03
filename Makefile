install:
	pip install --upgrade pip &&\
		pip install -r src/requirements.txt
		
		
format:
	black src/*.py

lint:
	pylint --disable=R,C src/nlp.py src/api.py src/client_api.py src/st_client_api.py

test:  
	python3.11 -m pytest -vv --cov=api test_api.py 
	
	
all: install lint test
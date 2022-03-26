VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

proto-gen-python: dep
	$(PYTHON) -m grpc_tools.protoc --proto_path=./proto/ ./proto/faker.proto --python_out=./gen/py/ --grpc_python_out=./gen/py/

proto-gen-go: $(PROTOC_GEN_GO)
	$(PROTOC) --go_out=. --go_opt=paths=import \
    --go-grpc_out=. --go-grpc_opt=paths=import \
    proto/faker.proto

proto-gen: proto-gen-python proto-gen-go

requirements:
	$(PIP) freeze > requirements.txt

venv:
	$(PYTHON) -m venv $(VENV)

dep: venv
	$(PIP) install -r requirements.txt

run: dep
	$(PYTHON) main.py

clean:
	 rm -rf __pycache__
	 rm -rf $(VENV)
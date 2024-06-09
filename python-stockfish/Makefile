.PHONY: all install dev-install uninstall clean

all: dev-install

install: clean
	pip install .

dev-install: clean
	pip install -e .

uninstall: clean
	pip uninstall pystockfish 
	$(RM) pystockfish_C.cpython-39-x86_64-linux-gnu.so

clean:
	$(RM) -rf __pycache__
	$(RM) -rf pystockfish/__pycache__
	$(RM) -rf build
	$(RM) -rf pystockfish.egg-info
	$(RM) -rf *.so
	$(RM) -rf dist/

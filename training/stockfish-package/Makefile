.PHONY: all install dev-install uninstall clean

all: dev-install

install: clean
	pip install .

dev-install: clean
	pip install -e .

uninstall: clean
	pip uninstall native_stockfish 
	$(RM) native_stockfish_C.cpython-39-x86_64-linux-gnu.so

clean:
	$(RM) -rf __pycache__
	$(RM) -rf native_stockfish/__pycache__
	$(RM) -rf build
	$(RM) -rf native_stockfish.egg-info
	$(RM) -rf *.so
	$(RM) -rf dist/

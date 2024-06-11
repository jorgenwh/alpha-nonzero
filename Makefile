.PHONY: all install dev-install uninstall clean

all: dev-install

install: clean
	pip install .

dev-install: clean
	pip install -e .

uninstall: clean
	pip uninstall anz
	$(RM) anz_C.cpython-39-x86_64-linux-gnu.so

clean:
	$(RM) -rf __pycache__
	$(RM) -rf anz/__pycache__
	$(RM) -rf build
	$(RM) -rf alpha_nonzero.egg-info
	$(RM) -rf *.so
	$(RM) -rf dist/

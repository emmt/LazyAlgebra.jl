SOURCES = \
    Project.toml \
    make.jl \
    src/index.md \
    src/introduction.md \
    src/mappings.md \
    src/refs.md \
    src/simplifications.md \
    src/vectors.md

default: build/index.html

clean:
	rm -f *~ */*~ */*/*~

force-rebuild:
	julia --color=yes make.jl

build/index.html: $(SOURCES)
	julia --color=yes make.jl

.PHOONY: default clean force-rebuild

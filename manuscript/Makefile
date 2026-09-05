BUILD_DIR := build
TARGET := thermodense-first-paper

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: main.tex references.bib figures/fig1.pdf figures/fig2.pdf figures/fig3.pdf figures/fig4.pdf figures/fig5.pdf figures/fig6.pdf
	mkdir -p $(BUILD_DIR)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) main.tex
	bibtex $(BUILD_DIR)/main
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) main.tex
	cp $(BUILD_DIR)/main.pdf $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET).pdf

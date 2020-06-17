# Tex files for writing thesis
The format of thesis and dissertation is defined by [NCKU](http://cid.acad.ncku.edu.tw/var/file/42/1042/img/3133795.pdf). The order in the cover page should be: school, department, master thesis or PhD dissertation, Chinese title, English title, student's name, advisor's name, year and month of defense. The order of whole thesis should be: certificate of defense, Chinese abstract, English abstract, acknowledgement, table of contents, list of tables, list of figures, nomenclature, main context, bibliography, appendix.

## Usage
1. Change the title and student name in the cover page to your thesis/dissertation title and your name.
2. When start writing, replace the command of dummy words in the main file with your context

## Required packages
* Texlive complier
    1. latex-cjk-all
    2. texlive-xetex
* Miktex compiler
    1. miktex-cjkutils
    2. xecjk
* (For Linux environment only) 
    1. [Windows fonts](https://www.ostechnix.com/install-microsoft-windows-fonts-ubuntu-16-04/)
    2. Kaiu font (標楷體)
        * Can be found in Fonts/
        * Copy to the directory where the font styles have been placed ```cp Fonts/kaiu.tff /usr/share/fonts/```
        * Make it available when loading fonts ```sudo fc-cache -f -v```
## Compile
* Normal case
    1. Use XeLaTex to compile the main file
    2. Compile with BibTex command for bibliography
* When nomenclature has been modified
    1. Use Makeindex command first and then XeLaTex
    2. If it doesn't change, repeat the previous step again

## Other available templates
1. [wengan-Li's template](https://github.com/wengan-li/ncku-thesis-template-latex)
2. [lycsjm's template](https://github.com/lycsjm/nckuthesis)
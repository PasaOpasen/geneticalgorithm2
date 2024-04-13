
#
# creates pdoc documentation
# 
# Notes:
#   must be run from project root
#   for this feature the python environment needs `pdoc` package
#

set -e

pkg=geneticalgorithm2

PYTHON="python"
for python in "venv/bin/python" ".venv/bin/python" 
do
    if [ -x "$(command -v ${python})" ]
    then
        echo -e "Use python from ${python}"
        PYTHON=${python}
        break
    else
        echo "Not found python from ${python}"
    fi
done

#
# rm old files
#
rm -rf ./documentation

#
# copy all module files to tmp dir
#
mkdir -p ./documentation/tmp
cp -Lr $pkg ./documentation/tmp/$pkg

#
# rm many unneseccary files (usually because of excess imports)
#
find ./documentation/tmp/$pkg -type f ! -name "*.py" -delete

# rm empty dirs finally
find ./documentation/tmp/$pkg -type d -empty -delete

#
# change some aliases
#
sed -i 's/np.ndarray/"np.ndarray"/g' ./documentation/tmp/$pkg/aliases.py

#
#
# DOCUMENTATION BUILD
#
#

#pdoc3 --html --force -o documentation ./documentation/tmp/dreamocr
${PYTHON} -m pdoc -d google -o documentation ./documentation/tmp/$pkg \
    --logo https://repository-images.githubusercontent.com/786023359/24dcc772-d337-433d-b82d-48b133900c59 \
    -e $pkg=https://github.com/PasaOpasen/geneticalgorithm2 \
    --footer-text "geneticalgorithm2 $(cat version.txt), ver. $(git describe --abbrev=0 --tags) + $(git rev-list --no-merges `git rev-list --tags --no-walk --max-count=1`..HEAD --count) commits"

# remove code files
rm -rf ./documentation/tmp

#xdg-open documentation/index.html

echo -e '\n\nSUCCESS\n\n' 

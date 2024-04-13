#!/bin/bash
set -eu -x

if [ ! -d documentation ]; then
    echo "Error: invalid directory *$PWD*. Deploy from repo root."
    ls -lah
    exit 1
fi

[ "$GH_PASSWORD" ] || exit 12

sitemap() {
    WEBSITE='https://pasaopasen.github.io/geneticalgorithm2'
    find -name '*.html' |
        sed "s,^\.,$WEBSITE," |
        sed 's/index.html$//' |
        grep -v '/google.*\.html$' |
        sort -u  > 'sitemap.txt'
    echo "Sitemap: $WEBSITE/sitemap.txt" > 'robots.txt'
}

head=$(git rev-parse HEAD)

git clone -b gh-pages "https://pasaopasen:$GH_PASSWORD@github.com/$GITHUB_REPOSITORY.git" gh-pages
mkdir -p gh-pages/docs
cp -R documentation/geneticalgorithm2/* gh-pages/docs/
cd gh-pages
sitemap
git add *
if git diff --staged --quiet; then
  echo "$0: No changes to commit."
  exit 0
fi

if ! git config user.name; then
    git config user.name 'github-actions'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
fi

git commit -a -m "CI: Update docs for ${GITHUB_REF#refs/tags/} ($head)"
git push

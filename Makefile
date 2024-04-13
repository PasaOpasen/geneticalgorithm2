
tag:
	git tag $(shell cat version.txt)
	git push --tags

pypipush:
	python setup.py develop
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/* --skip-existing

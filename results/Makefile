all: README.md charts-id/conclusion.png charts-en/conclusion.png

README.md: result_specs.json create_charts.py *.csv
		python3 create_charts.py

charts-id/conclusion.png: cmp.json create_charts.py benny-linux.csv benny-windows.csv
		python3 create_charts.py --lang id cmp cmp.json

charts-en/conclusion.png: cmp.json create_charts.py benny-linux.csv benny-windows.csv
		python3 create_charts.py --lang en cmp cmp.json


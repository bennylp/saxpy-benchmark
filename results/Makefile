all: README.md charts-id/conclusion.png charts-en/conclusion.png ../README.md

README.md: result_specs.json create_charts.py *.csv
		python3 create_charts.py

charts-id/conclusion.png: charts.json create_charts.py benny-linux.csv benny-windows.csv
		python3 create_charts.py --lang id cmp charts.json

charts-en/conclusion.png: charts.json create_charts.py benny-linux.csv benny-windows.csv
		python3 create_charts.py --lang en cmp charts.json

../README.md: ../front-page.md
		mv ../front-page.md ../README.md

../front-page.md: result_specs.json charts.json create_charts.py *.csv
		python3 create_charts.py report
	
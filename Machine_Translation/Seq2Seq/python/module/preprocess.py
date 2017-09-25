from bs4 import BeautifulSoup, SoupStrainer

def sgm_to_str(sgm_file, str_file):
	
	raw_data = open(sgm_file, 'r').read()
	f = open(str_file, 'w+')
	soup = BeautifulSoup(raw_data, 'lxml')
	for sample in soup.find_all('seg'):
		sen = sample.get_text()
		f.write(sen.strip() + "\n")
	f.close()





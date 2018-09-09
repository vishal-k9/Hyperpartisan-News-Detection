import xml.etree.ElementTree as ET
tree = ET.parse('ground-truth-training.xml')
root = tree.getroot()

with open("train_data.txt", "wb") as f:
	for article in root.findall('article'):
		idx= article.get('id')
		hp= article.get('hyperpartisan')
		bias= article.get('bias')
		url= article.get('url')
		f.write(idx+'\t'+url+'\t'+hp+'\t'+bias+'\n') 	

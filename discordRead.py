import codecs, os, time, datetime
from argparse import ArgumentParser

def isHeader(line):
	return len(line) >= 6 and '[' in line and ']' in line and line[-6] == '#'

def clean(line):
	if '\r' in line:
		line = line[:-2]
	elif '\n' in line:
		line = line[:-1]
	return line

def searchUser(data,path='\\source\\'):
	start = time.time()
	files = []
	found = []
	ids = []
	names = []
	savenames = []
	for u in data:
		ids.append(u[0])
		names.append(u[1])
		savenames.append(u[2])
		found.append([])
	dir = os.getcwd()
	for filename in os.listdir(dir+path):
		if filename[-4:] == '.txt':
			files.append(filename)
	#print(files)
	total = len(files)
	for i,filename in enumerate(files):
		print(i+1,total,filename)
		with codecs.open(dir+path+filename,'r',encoding='utf-8') as f:
			lines = f.readlines()
			# inside = []
			# for j in range(len(names)):
			# 	if ids[j] in lines:
			# 		inside.append(j)
			for i, line in enumerate(lines):
				if lines[i-1]=='\n' and i > 6:
					# for j in inside:
					for j in range(len(names)):
						if str(ids[j]) in line and names[j].casefold() in line.casefold():
							end = i+1
							while end < len(lines) and lines[end] != '\n':
								end += 1
							for k in range(i+1,end):
								l = clean(lines[k])
								found[j].append(l+'\n')
							found[j].append('\n')
	for j in range(len(names)):
		if savenames[j] != '':
			if not os.path.exists(dir+'/data/'+savenames[j]):
				os.makedirs(dir+'/data/'+savenames[j])
			os.chdir(dir+'/data/'+savenames[j])
		with codecs.open('input.txt','w',encoding='utf-8') as f:
			f.writelines(found[j])		
	os.chdir(dir)
	result = []
	for j in range(len(names)):
		result.append(len(found[j]))
	end = time.time()
	print('\nCompleted in {}'.format(datetime.timedelta(seconds=(end-start),microseconds=0))[:-4])
	return result

def cleanChannelTxt2(path='\\source\\'):
	start = time.time()
	path = os.getcwd()+path
	files = []
	for filename in os.listdir(path):
		if filename[-4:] == '.txt':
			files.append(filename)
	total = len(files)
	for i,filename in enumerate(files):
		print(i+1,total,filename)
		with codecs.open(os.path.join(path,filename),'r',encoding='utf-8') as f:
			cleanpath = os.path.join(path,'clean')
			if not os.path.isdir(cleanpath):
				os.makedirs(cleanpath)
			lines = f.readlines()[7:]
			new = []
			for i in range(2,len(lines)):
				if lines[i-1] == '\n':
					if isHeader(lines[i]):
						new.append(lines[i-1])
				elif isHeader(lines[i-1]):
					new.append(lines[i-1][lines[i-1].find(']')+2:-6]+' '+lines[i-1][-6:-1]+':\n')
				else:
					new.append(lines[i-1])
			with codecs.open(os.path.join(cleanpath,filename[:-4]+'-clean.txt'),'w',encoding='utf-8') as w:
				w.writelines(new)

				


# def cleanChannelTxt(path='\\source\\'):
# 	start = time.time()
# 	path = os.getcwd()+path
# 	files = []
# 	for filename in os.listdir(path):
# 		if filename[-4:] == '.txt':
# 			files.append(filename)
# 	total = len(files)
# 	for i,filename in enumerate(files):
# 		print(i+1,total,filename)
# 		with codecs.open(os.path.join(path,filename),'r',encoding='utf-8') as f:
# 			cleanpath = os.path.join(path,'clean')
# 			if not os.path.isdir(cleanpath):
# 				os.makedirs(cleanpath)
# 			name = filename.encode('ascii',errors='ignore').decode('ascii') #remove emoji
# 			w = codecs.open(os.path.join(cleanpath,name[:-4]+'-clean.txt'),'w',encoding='utf-8')
# 			new = []
# 			lines = f.readlines()
# 			pastHeader = False
# 			for j in range(len(lines)):
# 				if not pastHeader:
# 					if lines[j] == '\n':
# 						pastHeader = True
# 				else:
# 					if lines[j-1] == '\n' and (lines[j-2] == '\n' or lines[j-2 == '==============================================================']):
# 						# new.append(lines[j][lines[j].find(']')+2:-6]+':\n') #Removes timestamps and user ID #s for readability
# 						w.write(lines[j][lines[j].find(']')+2:-6]+':\n') #Removes timestamps and user ID #s for readability
# 						end = j+1
# 						while end < len(lines) and lines[end] != '\n':
# 							end += 1
# 						for k in range(j+1,end):
# 							# new.append(clean(lines[k]+'\n'))
# 							w.write(clean(lines[k]+'\n'))
# 						# new.append('\n')
# 						w.write('\n')
# 			# new = new[:-1]
# 			# cleanpath = os.path.join(path,'clean')
# 			# if not os.path.isdir(cleanpath):
# 			# 	os.makedirs(cleanpath)
# 			# name = filename.encode('ascii',errors='ignore').decode('ascii') #remove emoji
# 			# with codecs.open(os.path.join(cleanpath,name[:-4]+'-clean.txt'),'w',encoding='utf-8') as f:
# 			# 	f.writelines(new[:-1])
# 			 	# for line in lines:
# 				# 	f.write(line)
# 	end = time.time()
# 	print('\nCompleted in {}'.format(datetime.timedelta(seconds=(end-start),microseconds=0))[:-4])
				
# searchUser([['7777','holly','holly'],
# ['9501','silence','snake'],
# ['3141','pie','pie'],
# ['1276','mamba','mamba'],
# ['0568','math','math'],
# ['0601','xen0','sayoriBV'],
# ['3500','lenni','lenni'],
# ['8984','stick','sam'],
# ['5179','chivi','chivi'],
# ['1878','ck','nick'],
# ['0736','infinity pain','smol'],
# ['4615','harold','harold'],
# ['0739','cadence','cadence'],
# ['3596','uzer','uzer'],				
# ['9905','toli','tal'],
# ['5816','pepe','pepe'],
# ['3898','can','can'],
# ['5816','pepe','pepe'],
# ['6685','dezi','dezi'],
# ['2362','water','water'],
# ['5725','noctsuke','noctsuke']])

# searchUser([['9905','toli','talvent'],
# ['3500','lenni','lennivent'],
# ['5816','pepe','pepevent']])

searchUser([['9501','radio','snake']])

# cleanChannelTxt()
# cleanChannelTxt2()
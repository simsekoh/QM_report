import json as js
import pandas as pd
import numpy as np
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns

import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn import metrics
import time

from collections import Counter
print('opening files..')
#Files:
#--------------------------------------------------------------------------------------------
# read file
with open('./Rawdata/anonymized_project.json', 'r') as json_file:
    data=json_file.read()


#read references
with open('../Rawdata/references.json','r') as json_file2:
	data2= json_file2.read()

# parse files
obj = js.loads(data)
refs = js.loads(data2)

#open result.txt
f = open('./results.txt', 'w')

#write header for results.txt
f.write('\t\tQuality  Match GmbH - bicycle project crowd evaluation\n')
f.write('\t\t\t\tAuthor: Simon Kohlhepp\n')
#--------------------------------------------------------------------------------------------
print('initialize variables')
#Variables:
#--------------------------------------------------------------------------------------------
#initialize variables

annotationtime_acc = 0
annotationtime_max = 0
annotationtime_min = 0
annotationtime_avg = 0

count_obj=0
count_ans=0
count_yes = 0
ref_set_count = 0
count_disagree_1 = 0
count_disagree_2 = 0
count_disagree_3 = 0
count_disagree_4 = 0
count_disagree_5 = 0
count_disagree_6 = 0
count_true_positive = 0
count_true_negative = 0
count_false_negative = 0
count_false_positive = 0
#--------------------------------------------------------------------------------------------

#Dictionaries, Lists & Sets
#--------------------------------------------------------------------------------------------
# sets of all occuring images/annotators -> used as key container/ for iterating purposes
images = set()
annotators = set()

#list will contain all annotators iin ascending order
annotators_sorted = []

#list of all annotations -> used to determine the annotation count for each annotator
annotators_all_annotations = []

#list containing the annotated 'yes'-rate for each image 
images_ans = []

#list will contain every workpackage where each consists of ~10 annotations
bundles= []

# first list will contain the 2nd and 3rd list zipped such that for any "cant_solve" flag the annotator and the corresponding image is saved
cant_solve = []
user_cant_solve = []
img_cant_solve = []

# first list will contain the 2nd and 3rd list zipped such that for any "corrupt" flag the annotator and the corresponding image is saved
corrupt = []
user_corrupt = []
img_corrupt = []

#dictionary for amount of TP, FP , TN, FN, total duration & annotation count of each annotator -> used for metrics
annotators_contingency = {}

#lists for contingency matrix
results = []
predicted = []

#lists for plotting purposes
pie_chart_data = []
pie_chart_labels = []
annotators_duration = []

#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Program start
#--------------------------------------------------------------------------------------------
print('start computation -> this could take several seconds')
#--------------------------------------------------------------------------------------------
#count 'yes' annotations occuring in the reference set, used to determine if the set is balanced
for i in refs:
	if(refs[i]['is_bicycle'] == True):
		ref_set_count+=1
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#compute durations & collect corrupt/cant_solve images/annotations
#create dictionary "bundles" consisting out of sextuples of data for each annotation submitted. 
for wk_pkg in obj['results']['root_node']['results']:
	
	#increase count of workpackages
	count_obj = count_obj+1
	
	#initialize local 1-dimensional dictionaries for each relevant date
	answer = []
	vendor_user_id = []
	hash_id= []
	validity=[]
	img_url = []
	duration = []
	current_bundle = []
	
	#fill dictionaries with data
	for task in obj['results']['root_node']['results'][wk_pkg]['results']:
		
		#increase count of given answers
		count_ans = count_ans+1
		
		#add data to correspoding list
		answer.append(task['task_output']['answer'])
		vendor_user_id.append(task['user']['vendor_user_id'])
		hash_id.append(wk_pkg)
		img_url.append(task['task_input']['image_url'])
		duration.append(task['task_output']['duration_ms'])
		
		#since "images" is a set it won't add duplicates
		images.add(str(task['task_input']['image_url'])[65:73])
		
		#since "annotators" is a set it won't add duplicates
		annotators.add(task['user']['vendor_user_id'])
		
		#add "user_id" element for each annotation made
		annotators_all_annotations.append(task['user']['vendor_user_id'])
		
		#the 'validity'-flag is used to determine if any of the cant_solve/corrupt-flags was set.
		if ((task['task_output']['cant_solve'] ==True or task['task_output']['corrupt_data']) == True):
			validity.append(True)
		else:
			validity.append(False)
			
		# incease counter if current answer is 'yes'
		if (task['task_output']['answer'] == "yes"):
			count_yes=count_yes+1	
			
		#collect images and user if 'cant_solve'-flag was set 
		if (task['task_output']['cant_solve'] == True):
			
			user_cant_solve.append(task['user']['vendor_user_id'])
			img_cant_solve.append(str(task['task_input']['image_url'])[65:73])
		
		#collect images and user if 'corrupt_data'-flag was set 
		if (task['task_output']['corrupt_data'] == True):
			
			user_corrupt.append(task['user']['vendor_user_id'])
			img_corrupt.append(str(task['task_input']['image_url'])[65:73])
		
		#---------------Annotationtimes----------------------------
		annotationtime_acc += task['task_output']['duration_ms']
		
		
		if( task['task_output']['duration_ms'] > annotationtime_max and task['task_output']['cant_solve'] == False and task['task_output']['corrupt_data'] == False):
			annotationtime_max = task['task_output']['duration_ms']
			
		if( (task['task_output']['duration_ms'] <= annotationtime_min and task['task_output']['duration_ms'] > 0) or annotationtime_min ==0):
			annotationtime_min = task['task_output']['duration_ms']	
		#----------------------------------------------------------
		
		
		
	#zip data into 1 dictionary such that each item consists of a sextuple values of the same index 
	current_bundle = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(hash_id,img_url,answer,vendor_user_id,duration,validity)]
	
	#insert current annotation bundle
	bundles.append(current_bundle)
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------		
#zip collected image/user combination into list of tuples
cant_solve = [[a,b] for a,b in zip(user_cant_solve, img_cant_solve)]
corrupt = [[a,b] for a,b in zip(user_corrupt, img_corrupt)]
#sort annotators 
annotators_sorted = sorted(annotators)
#--------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------

for i in range(len(images)):
	images_ans_count=0
	images_ans_count_yes = 0
	
	for cur in bundles[i]:
		images_ans_count += 1
		
		if( cur[2] == 'yes'):
			images_ans_count_yes+=1
			
	yes_rate = images_ans_count_yes/images_ans_count
	
	if (yes_rate == 0.5):
		count_disagree_1 += 1
	elif(yes_rate == 0.4 or yes_rate == 0.6):
		count_disagree_2 +=1
	elif(yes_rate == 0.3 or yes_rate == 0.7):
		count_disagree_3 +=1
	elif(yes_rate == 0.2 or yes_rate == 0.8):
		count_disagree_4 +=1
	elif(yes_rate == 0.1 or yes_rate == 0.9):
		count_disagree_5 +=1
	else:
		count_disagree_6 += 1
		
	images_ans.append(yes_rate)
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#
cant_solve.sort()
cant_solve_counts = Counter(x[0] for x in cant_solve)
#
corrupt.sort()
corrupt_counts = Counter(x[0] for x in corrupt)
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
# initialize dictionary
for pers in annotators:
	annotators_contingency[pers] = {'count_true_positive':0, 'count_true_negative':0, 'count_false_positive':0, 'count_false_negative':0, 'total_duration':0, 'count_annotated':0}
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#iterate through all annotations to determine the TP, FP, TN, FN Counts of each annotator
for bundle in bundles:
	for task in bundle:
		if(task[4] >= 0):
			annotators_contingency[task[3]]['count_annotated'] += 1
			annotators_contingency[task[3]]['total_duration'] += task[4]
			
		if(refs[str(str(task[1])[65:73])]['is_bicycle'] == True and task[2] == 'yes'): 
			annotators_contingency[task[3]]['count_true_positive'] += 1
			count_true_positive += 1
			
		elif ((refs[str(str(task[1])[65:73])]['is_bicycle'] == False and task[2] == 'no')):
			annotators_contingency[task[3]]['count_true_negative'] += 1
			count_true_negative += 1
				
		elif((refs[str(str(task[1])[65:73])]['is_bicycle'] == True and task[2] == 'no')):
			annotators_contingency[task[3]]['count_false_negative'] += 1
			count_false_negative += 1
				
		else:
			annotators_contingency[task[3]]['count_false_positive'] += 1	
			count_false_positive += 1
			
		#----for plotting CM--------		
		if	(task[2] == 'yes'):
			results.append(True)
		else:
			results.append(False)			
		predicted.append(refs[str(str(task[1])[65:73])]['is_bicycle'])
#--------------------------------------------------------------------------------------------
print('computing values done')
#--------------------------------------------------------------------------------------------
# Write to file
#--------------------------------------------------------------------------------------------
print('writing results to file...')
#--------------------------------------------------------------------------------------------
f.write('\t\t--------------------------------------\n')
f.write(str(count_obj)+" Datasets available\n")
f.write(str(len(annotators_all_annotations))+" annotations\n")
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------
f.write('Task 1:\n')
f.write('\ta)\t')
f.write(str(len(annotators))+' annotators participated\n')
f.write('\tb)\t')
f.write('durations: 	\n')
f.write('\t\t\tmax: '+str(annotationtime_max)+'\n')
f.write('\t\t\tmin: '+str(annotationtime_min)+'\n')
f.write('\t\t\tavg: '+str(annotationtime_acc/count_ans)+'\n')
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------
f.write('\t\tAverage duration per annotator: 	\n')
for i in annotators_sorted:
	f.write('\t\t\t'+str(i)+'\t'+str(round(annotators_contingency[i]['total_duration']/annotators_contingency[i]['count_annotated'], 2))+' ms\n')
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------
f.write('\tc)\t')
f.write('Annotations per annotator: 	\n')
for i in annotators_sorted:
	f.write('\t\t\t'+str(i)+'\t'+str(annotators_all_annotations.count(i))+'\n')
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------
f.write('\td)\t')
f.write('high disagreement: 	\n')
f.write('\t\t\t100% disagreement rate: \t\t'+str(count_disagree_1)+' images\n')
f.write('\t\t\t 80% disagreement rate: \t\t'+str(count_disagree_2)+' images\n')
f.write('\t\t\t 60% disagreement rate: \t\t'+str(count_disagree_3)+' images\n')
f.write('\t\t\t 40% disagreement rate: \t\t'+str(count_disagree_4)+' images\n')
f.write('\t\t\t 20% disagreement rate: \t\t'+str(count_disagree_5)+' images\n')
f.write('\t\t\t  0% disagreement rate: \t\t'+str(count_disagree_6)+' images\n')
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------
f.write('Task 2:\n')
f.write('\ta)\t')
f.write('Flags used: 	\n')
f.write('\t\tFlag: can\'t solve: '+str(len(cant_solve))+' appearances in total\n')
f.write('\t\tappearance per annotator: \n')
for x in dict(cant_solve_counts.most_common()):
	f.write('\t\t\t'+str(x)+'\t'+str(cant_solve_counts[x])+'\n')
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------
f.write('\t\tFlag: corrupt data: '+str(len(corrupt))+' appearances in total\n')
f.write('\t\tappearance per annotator: \n')
for x in dict(corrupt_counts.most_common()):
	f.write('\t\t\t'+str(x)+'\t'+str(corrupt_counts[x])+'\n')	
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------	
f.write('Task 3:\n')
f.write('\t\tIs the reference set balanced?\n')
if((ref_set_count/len(refs)) >0.49 and ((ref_set_count/len(refs)) < 0.51)):
	f.write(str(ref_set_count)+" yes\n")
	f.write(str(len(refs)-ref_set_count)+" no\n")
	f.write("-->"+str(ref_set_count/len(refs))+" %\n")
	f.write("--->  set is balanced\n")
else:
	f.write(str(ref_set_count)+" yes\n")
	f.write(str(len(refs)-ref_set_count)+" no\n")
	f.write("-->"+str(ref_set_count/len(refs))+" %\n")
	f.write("--->  set is not balanced\n")		
f.write('\t\t--------------------------------------\n')	
#--------------------------------------------------------------------------------------------
f.write('Task 4:')
f.write('\t\tPerformance of the whole crowd: \n')
f.write('\t'+'Predictions total: '+str((count_true_positive+count_true_negative+count_false_positive+count_false_negative))+'\n')
f.write('\t'+'Accuracy: '+str((count_true_positive+count_true_negative)/(count_true_positive+count_true_negative+count_false_positive+count_false_negative))+'\n')
f.write('\t'+'Precision: '+str(count_true_positive/(count_true_positive+count_false_positive))+'\n')
f.write('\t'+'Recall/Sensitivity: '+str(count_true_positive/(count_true_positive+count_false_negative))+'\n')
f.write('\t'+'F1-Score: '+str((2*count_true_positive)/((2*count_true_positive)+count_false_negative+count_false_positive))+'\n')
f.write('\t'+'Specificity: '+str(count_true_negative/(count_false_positive+count_true_negative))+'\n')
f.write('\t'+'MCC: '+str(((count_true_positive*count_true_negative)-(count_false_positive*count_false_negative))/(sqrt((count_true_positive+count_false_positive)*(count_true_positive+count_false_negative)*(count_true_negative+count_false_positive)*(count_true_negative+count_false_negative))))+'\n')
f.write('\t\t--------------------------------------\n')	
f.write('\t\tPerformance of each annotator: \n')
f.write('\t______\n')
#--------------------------------------------------------------------------------------------
mcc_rates = []
duration = []
count_annotated = []
for i in annotators_sorted:
	f.write('\t'+str(i)+'\n')
	f.write('\t'+'Total annotated: '+'\t'+str(annotators_contingency[i]['count_annotated'])+' images\n')
	f.write('\t'+'Average duration: '+'\t'+str( annotators_contingency[i]['total_duration']/annotators_contingency[i]['count_annotated'])+' ms\n')
	f.write('\t'+'Accuracy: '+'\t\t\t'+str((annotators_contingency[i]['count_true_positive']+annotators_contingency[i]['count_true_negative'])/(annotators_contingency[i]['count_true_positive']+annotators_contingency[i]['count_true_negative']+annotators_contingency[i]['count_false_positive']+annotators_contingency[i]['count_false_negative']))+'\n')
	f.write('\t'+'Precision: '+'\t\t\t'+str((annotators_contingency[i]['count_true_positive']/(annotators_contingency[i]['count_true_positive']+annotators_contingency[i]['count_false_positive'])))+'\n')
	f.write('\t'+'Recall/Sensitivity: '+str((annotators_contingency[i]['count_true_positive']/(annotators_contingency[i]['count_true_positive']+annotators_contingency[i]['count_false_negative'])))+'\n')
	f.write('\t'+'Specificity: '+str((annotators_contingency[i]['count_true_negative']/(annotators_contingency[i]['count_false_positive']+annotators_contingency[i]['count_true_negative'])))+'\n')
	f.write('\t'+'F1-Score: '+'\t\t\t'+str((2*(annotators_contingency[i]['count_true_positive']))/(2*(annotators_contingency[i]['count_true_positive'])+annotators_contingency[i]['count_false_negative']+annotators_contingency[i]['count_false_positive']))+'\n')
	f.write('\t'+'MCC: '+'\t\t\t'+str(((annotators_contingency[i]['count_true_positive']*annotators_contingency[i]['count_true_negative'])-(annotators_contingency[i]['count_false_positive']*annotators_contingency[i]['count_false_negative']))/(sqrt((annotators_contingency[i]['count_true_positive']+annotators_contingency[i]['count_false_positive'])*(annotators_contingency[i]['count_true_positive']+annotators_contingency[i]['count_false_negative'])*(annotators_contingency[i]['count_true_negative']+annotators_contingency[i]['count_false_positive'])*(annotators_contingency[i]['count_true_negative']+annotators_contingency[i]['count_false_negative']))))+'\n')
	f.write('\t______\n')
	count_annotated.append(annotators_contingency[i]['count_annotated'])
	duration.append(annotators_contingency[i]['total_duration']/annotators_contingency[i]['count_annotated'])
	#append mcc rates corresponding to each annotator
	mcc_rates.append(((annotators_contingency[i]['count_true_positive']*annotators_contingency[i]['count_true_negative'])-(annotators_contingency[i]['count_false_positive']*annotators_contingency[i]['count_false_negative']))/(sqrt((annotators_contingency[i]['count_true_positive']+annotators_contingency[i]['count_false_positive'])*(annotators_contingency[i]['count_true_positive']+annotators_contingency[i]['count_false_negative'])*(annotators_contingency[i]['count_true_negative']+annotators_contingency[i]['count_false_positive'])*(annotators_contingency[i]['count_true_negative']+annotators_contingency[i]['count_false_negative']))))

#--------------------------------------------------------------------------------------------	
f.write('NOTE: annotator_19 had 5 negative duration_ms values. These annotations have been eliminated before computing the scores.\n\n')
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------
#create list of tuples 
user_rating = [[a,b] for a,b in zip(mcc_rates,annotators_sorted)]
user_rating2 = [[a,b] for a,b in zip(duration,annotators_sorted)]
user_rating3 = [[a,b] for a,b in zip(count_annotated,annotators_sorted)]

user_rating.sort(reverse=True)
user_rating2.sort()
user_rating3.sort(reverse=True)

#Rank users by Matthews correlation coefficient
f.write('Annotators ranked by Matthews correlation coefficient\n')
for tupl in user_rating:
	f.write('\t\t\t'+str(tupl[0])+'\t'+str(tupl[1])+'\n')
f.write('\t\t--------------------------------------\n')
#Rank users by average duration
f.write('Annotators ranked by average duration\n')
for tupl in user_rating2:
	f.write('\t\t\t'+str(tupl[0])+'\t'+str(tupl[1])+'\n')
f.write('\t\t--------------------------------------\n')
#Rank users by amount of annotations
f.write('Annotators ranked by amount of annotations\n')
for tupl in user_rating3:
	f.write('\t\t\t'+str(tupl[0])+'\t'+str(tupl[1])+'\n')
f.write('\t\t--------------------------------------\n')
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# plotting
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
print('writing plots to \'Plot\' folder -> this could take several seconds')
#--------------------------------------------------------------------------------------------
#plot bar chart frequency distribution
dtf= pd.DataFrame(images_ans)

ax = dtf.value_counts().sort_index().plot(kind="bar")
totals= []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for p in ax.patches:
     ax.annotate(str(round((p.get_height()*100/9087),2))+'%', (p.get_x() , p.get_height()+20))

ax.grid(axis="y")
plt.suptitle("'yes'-rate frequency distribution", fontsize=20)
#plt.subplots_adjust(top=0.966,bottom=0.17,left=0.094,right=0.963,hspace=0.2,wspace=0.2)
plt.savefig('./Plots/FD_complete.png', dpi=300)
#plt.show()
plt.close()
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
# plot contingency matrix
classes = [True, False]

fig, ax = plt.subplots()
cm = metrics.confusion_matrix(results, predicted, labels=classes)

sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
ax.set(xlabel="Annotated", ylabel="Reference", title="Confusion matrix")
ax.set_yticklabels(labels=classes, rotation=0)
ax.set_xticklabels(labels=classes, rotation=0)
plt.savefig('./Plots/CM_complete.png', dpi=300)
#plt.show()
plt.close()
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
#plot contingency matrix for each annotator/ collect data for pie chart
for i in annotators:
	pie_chart_labels.append(i)
	pie_chart_data.append(annotators_contingency[i]['count_annotated'])
	annotators_duration.append(annotators_contingency[i]['total_duration']/annotators_contingency[i]['count_annotated'])
	classes = [ True, False]
	fig, ax = plt.subplots()
	cm = [[annotators_contingency[i]['count_true_positive'], annotators_contingency[i]['count_false_negative'] ],[annotators_contingency[i]['count_false_positive'],annotators_contingency[i]['count_true_negative']]]
	sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
	ax.set(xlabel="Annotated", ylabel="Reference", title='Confusion matrix '+str(i))
	ax.set_yticklabels(labels=classes, rotation=0)
	ax.set_xticklabels(labels=classes, rotation=0)
	plt.savefig('./Plots/CM_'+str(i)+'.png')
	#plt.show()
	plt.close()
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#plot bar chart average durations
dtf= pd.DataFrame(annotators_duration, annotators)
ax = dtf.sort_values(0).plot(kind="bar", width=0.8)
totals= []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for p in ax.patches:
     ax.annotate(str(round(p.get_height(),2)), (p.get_x() , p.get_height()+20))
ax.grid(axis="y")
ax.get_legend().remove()

plt.suptitle("annotation durations: average", fontsize=20)

plt.savefig('./Plots/AD_complete.png', dpi=300)
plt.show()
plt.close()
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#plot pie chart annotation amount
colors = sns.color_palette('pastel')[0:22]
plt.pie(pie_chart_data, labels = pie_chart_labels, colors = colors, autopct='%.0f%%')
#plt.show()
plt.savefig('./Plots/AA_Complete', dpi=300)
#--------------------------------------------------------------------------------------------

f.close()
print("Done.")
exit()

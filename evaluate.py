fname1 = './testData/label.txt'
fname2 = './testData/score.txt'

with open(fname1) as data:
    label = [[float(i) for i in line.split()] for line in data.readlines()]

with open(fname2) as data:
    score = [[float(i) for i in line.split()] for line in data.readlines()]


print('label data')
print(label)
print('')

print('score data')
print(score)
print('')

print('calculate iou')

x_temp = [0,0,0,0];
y_temp = [0,0,0,0];

x_temp[0] = label[0][0];
x_temp[1] = label[0][2];
y_temp[0] = label[0][1];
y_temp[1] = label[0][3];

for i in range(0,27):
	x_temp[2] = score[i][0];
	x_temp[3] = score[i][2];
	y_temp[2] = score[i][1];
	y_temp[3] = score[i][3];
	print(x_temp);
	print(y_temp);

	label_x = x_temp[1] - x_temp[0];
	label_y = y_temp[1] - y_temp[0];
	label_sum = label_x*label_y

	score_x = x_temp[3] - x_temp[2];
	score_y = y_temp[3] - y_temp[2];
	score_sum = score_x*score_y

	max_sum = label_sum + score_sum

	x_temp.sort();
	print(x_temp);
	y_temp.sort();
	print(y_temp);

	inner_x = x_temp[2] - x_temp[1];
	inner_y = y_temp[2] - y_temp[1];
	inner_sum = inner_x * inner_y
	print(inner_sum)

	full_size = max_sum - inner_sum

	print('iou value')
	iou = inner_sum / full_size
	print(iou)




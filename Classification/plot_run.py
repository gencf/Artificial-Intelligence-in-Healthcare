import matplotlib.pyplot as plt

train = []
test = []
r = [50,100,160,200,230,256,280,300]
names = ["resize 50.png",
"resize 100.png",
"resize 160.png",
"resize 200.png",
"resize 230.png",
"resize 256.png",
"resize 280.png",
"resize 300.png",
]
f = open("run9.txt","r")

kontrol = 0
for i,line in enumerate(f.readlines()):
    if line[:3]=="Acc":
        if kontrol%2==0:
            train.append(float(line[10:15]))
        else:    
            test.append(float(line[10:15]))
        
        kontrol += 1
#plt.plot(range(1,16),train)
#plt.plot(range(1,16),test)

for i in range(8):
    plt.figure(i)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(names[i])
    plt.plot(range(15),train[i*15:(i+1)*15])
    plt.plot(range(15),test[i*15:(i+1)*15])
    #plt.savefig(names[i])

plt.show()

f.close()

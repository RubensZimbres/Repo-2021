import numpy as np
from itertools import permutations
from collections import OrderedDict

outcome=[]
with open('/home/anaconda3/work/rockyou_smol.txt','r', encoding="latin-1") as f:
    my_list = list(f)
    my_list = [x.rstrip() for x in my_list] 
    my_list=np.array(my_list).astype(str)
    for word in my_list[0:4]:
        try:
            outcome.append(word.replace("o","0"))
            outcome.append(word.replace("i","1"))
            outcome.append(word.replace("z","2"))
            outcome.append(word.replace("s","$"))
            outcome.append(word.replace("e","3"))
            outcome.append(word.replace("a","@"))
            outcome.append(word.replace("g","9"))
            outcome.append(word.replace("o","0").replace("i","1").replace("z","2").replace("s","$").replace("e","3").replace("a","@").replace("g","9"))

            outcome.append(word.title().replace("o","0"))
            outcome.append(word.title().replace("i","1"))
            outcome.append(word.title().replace("z","2"))
            outcome.append(word.title().replace("s","$"))
            outcome.append(word.title().replace("e","3"))
            outcome.append(word.title().replace("a","@"))
            outcome.append(word.title().replace("g","9"))
            outcome.append(word.title().replace("o","0").replace("i","1").replace("z","2").replace("s","$").replace("e","3").replace("a","@").replace("g","9"))

            outcome.append(word+'+')
            outcome.append(word+'!')
            outcome.append(word+'&')
            outcome.append(word+'#')
            outcome.append(word+'-')
            outcome.append(word+'%')
            outcome.append(word.title())
            outcome.append(word.title()+'+')
            outcome.append(word.title()+'!')
            outcome.append(word.title()+'&')
            outcome.append(word.title()+'#')
            outcome.append(word.title()+'-')
            outcome.append(word.title()+'%')

            word1 = [''.join(p) for p in permutations(word.title())][-1]
            outcome.append('+'+word1)
            outcome.append('!'+word1)
            outcome.append('&'+word1)
            outcome.append('#'+word1)
            outcome.append('-'+word1)
            outcome.append('%'+word1)
            outcome.append('+'+word1)
            outcome.append('!'+word1)
            outcome.append('&'+word1)
            outcome.append('#'+word1)
            outcome.append('-'+word1)
            outcome.append('%'+word1)
            outcome.append(word1.replace("o","0"))
            outcome.append(word1.replace("i","1"))
            outcome.append(word1.replace("z","2"))
            outcome.append(word1.replace("s","5"))
            outcome.append(word1.replace("e","3"))
            outcome.append(word1.replace("a","@"))
            outcome.append(word1.replace("g","9"))
            outcome.append(word1.replace("o","0").replace("i","1").replace("z","2").replace("s","$").replace("e","3").replace("a","@").replace("g","9"))

            outcome.append(word1.title().replace("o","0"))
            outcome.append(word1.title().replace("i","1"))
            outcome.append(word1.title().replace("z","2"))
            outcome.append(word1.title().replace("s","$"))
            outcome.append(word1.title().replace("e","3"))
            outcome.append(word1.title().replace("a","@"))
            outcome.append(word1.title().replace("g","9"))
            outcome.append(word1.title().replace("o","0").replace("i","1").replace("z","2").replace("s","$").replace("e","3").replace("a","@").replace("g","9"))
            
        except:
            pass


with open("./rockyou_improved_6Mi.txt", 'w') as f:
    f.write("\n".join(map(str, list(OrderedDict.fromkeys(outcome))
)))

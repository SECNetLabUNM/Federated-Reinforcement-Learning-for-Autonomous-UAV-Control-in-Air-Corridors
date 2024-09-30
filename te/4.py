class a():
    var=[]
    def __init__(self):
        self.term = False


class b(a):
    def __init__(self):
        super().__init__()
        self.b = 1

    def p(self):
        print(self.term)

class c(a):
    def __init__(self):
        self.c=1
a.var=5
b_instance=b()
print(b_instance.var)
# c_instance=c()
# # b_instance.var=3
# # print(type(b))
# # print(b.var)
# b.var.append(3)
# print(a.var)
# print(c.var)
# b.var=[]
# print(a.var)
# print(c.var)

import re

#Match phone numbers. Return True or False. See README for details.
def problem1(searchstring):
    ans = re.search("^(\(\d\d\d\)\s|\d\d\d-)?\d\d\d-\d\d\d\d$",searchstring)
    if ans is None:
        return False
    else:
        return True
#Extract street name from address. See README for details.
def problem2(searchstring) :
    ans = re.search(r"(.*?)(\d+) (.*?) (\w+\.)(.*?)",searchstring)
##  print(ans)
    if ans:
        return ans.group(3)
    
#Garble street name. See README for details
def problem3(searchstring) :
    ans = re.search(r"(.*?)(\d+) (.*?) (\w+\.)(.*?)",searchstring)
    return ans.group(1)+ans.group(2)+" "+ans.group(3)[::-1]+" "+ans.group(4)+ans.group(5)
        
if __name__ == '__main__' :
    print(problem1('765-494-4600')) #True
    print(problem1('765-494-4600 ')) #False
    print(problem1(' (765) 494 4600')) #False
    print(problem1('(765) 494-4600')) #True    
    print(problem1('494-4600')) #True
    
    print(problem2('The EE building is at 465 Northwestern Ave.')) #Northwestern
    print(problem2('Meet me at 201 South First St. at noon')) #South First
    
    print(problem3('The EE building is at 465 Northwestern Ave.'))
    print(problem3('Meet me at 201 South First St. at noon'))

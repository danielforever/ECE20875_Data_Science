

def histogram(data, n, l, h):
    if (n<=0 or h < l or isinstance(n, int) is False):
        print("n is not good")
        return []
    ans = [0] * n
    w = (h-l)/n
    for i in data:
        if (i > l and i <h):
            bin = (i-l)/w
            ans[int(bin)]=ans[int(bin)]+1
    return ans

def addressbook(name_to_phone, name_to_address):
    address_to_all=dict()
    for x,y in name_to_address.items():
        if y not in address_to_all:
            address_to_all[y]=None
        if address_to_all[y] is None:
            address_to_all[y]=([x],name_to_phone[x])
        else:
            address_to_all[y][0].append(x)
            if name_to_phone[x] is not address_to_all[y][1]:
                print("Warning: "+ x + " has a different number for "+y+" than "+address_to_all[y][0][0]+". Using the number for "+address_to_all[y][0][0]+".")
    return address_to_all







#if __name__ == "__main__":
    #data = [-2, -2.2, 0, 5.6, 8.3, 10.1, 30, 4.4, 1.9, -3.3, 9, 8]
    #hist = histogram(data,10, -5, 10)
    #print(hist)
    #name_to_phone = {'alice': 5678982231, 'bob': '111-234-5678', 'christine': 5556412237, 'daniel': '959-201-3198',
    #                 'edward': 5678982231}
    #name_to_address = {'alice': '11 hillview ave', 'bob': '25 arbor way', 'christine': '11 hillview ave',
    #                  'daniel': '180 ways court', 'edward': '11 hillview ave'}
    #address_to_all = addressbook(name_to_phone, name_to_address)
    #print(address_to_all)


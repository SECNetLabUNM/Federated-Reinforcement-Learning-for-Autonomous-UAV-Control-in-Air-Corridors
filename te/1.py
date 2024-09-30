s = "A man, a plan, a canal: Panama"
s=list(s.lower())
a=[i for i in s if ord('a')<=ord(i)<=ord('z')]
print(a==a[::-1])


# left, right = 0, len(s) - 1
#
# while left < right:
#     if cleaned_s[left] != cleaned_s[right]:
#         remove_left = clean_s[:left] + clean_s[left+1:]
#         remove_right = clean_s[:right] + clean_s[right+1:]
#
#         return remove_left == remove_left[::-1] or remove_right == remove_right[::-1]
#
#     left += 1
#     right -= 1
#
# return True

s='abcd'
left=1
print(s[:left]+s[left+1:])

print(s[1:3]==s[:])
print(s[:5]==s[:])
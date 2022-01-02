from cpython cimport array
import array

cpdef array.array intersect_two_doc_lists(array.array list1, array.array list2):
    cdef int len1 = len(list1)
    cdef int len2 = len(list2)
    cdef int pos1 = 0
    cdef int pos2 = 0
    cdef array.array result = array.array("Q")

    while pos1 < len1 and pos2 < len2:
        if list1[pos1] == list2[pos2]:
            result.append(list1[pos1])
            pos1 += 1
            pos2 += 1
        elif list1[pos1] < list2[pos2]:
            pos1 += 1
        else:
            pos2 += 1

    return result


cpdef int tokens_contain(list large, list small):
    cdef int len_large = len(large)
    cdef int len_small = len(small)
    cdef int j = 0
    cdef int k = 0

    while j < len_large:
        if small[k] == large[j].id:
            k += 1
            j += 1
            if k == len_small:
                return True
        elif k == 0:
            j += 1
        else:
            k = 0
    return False
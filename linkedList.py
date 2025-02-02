# Author: lyy1119 , Github
# Linkedlist

class LinkedList:
    class Node:
        def __init__(self , item=None):
            self.item = item
            self.next = None

        def __str__(self):
            return f"{self.item}"

        def __repr__(self):
            return f"<type:{type(self.item)},content:{self.item}>"

    class LinkedListIterator:
        def __init__(self, head):
            self.current = head

        def __iter__(self):
            return self

        def __next__(self):
            if not self.current:
                raise StopIteration
            item = self.current.item
            self.current = self.current.next
            return item

    def __init__(self):
        self.head = None
        self.tail = self.head
        self.length = 0

    def __iter__(self):
        return self.LinkedListIterator(self.head)

    def __str__(self):
        str = '<HEAD>'
        if self.is_empty():
            pass
        else:
            for node in self:
                str += f" --> {node}"
        str += " --> <TAIL>"
        return str

    def __len__(self):
        return self.length

    def __getitem__(self , index):
        if isinstance(index , int):
            if index < 0: # 处理负数索引
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("LinkedList index out of range.")
            current = self.head
            for _ in range(index):
                current = current.next
            return current
        elif isinstance(index , slice):
            start , stop , step = index.indices(len(self) + 1)
            print(start , stop , step)
            if step < 0:
                raise IndexError("Reverse slice is not support.")
            result = LinkedList() # 返回结果
            # 移动到开始出
            current = self.head
            for _ in range(start):
                current = current.next
            # 按给定步长迭代
            for _ in range(start , stop , step):
                result.append(current)
                # 按照步长移动
                for _ in range(step):
                    current = current.next
            return result

    def append(self , element):
        newNode = self.Node(element)
        if self.is_empty(): # 处理第一个加入的元素
            self.head = newNode
            self.tail = newNode
        else:
            self.tail.next = newNode
            self.tail = newNode
        self.length += 1

    def extend_by_list(self , li: list) -> None:
        for i in li:
            self.append(i)

    def is_empty(self):
        return self.head == None
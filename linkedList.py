# Author: lyy1119 , Github
# Linkedlist

class Node:
    def __init__(self , item=None):
        self.item = item
        self.next = None
        pass

class LinkedList:
    def __init__(self):
        self.head = Node()
        self.tail = self.head
        pass

    def append(self , element):
        newNode = Node()
        newNode.item = element
        self.tail.next = newNode
        self.tail = newNode

    def print(self):
        nowNode = self.head.next
        while nowNode != None:
            print(nowNode.item)
            nowNode = nowNode.next

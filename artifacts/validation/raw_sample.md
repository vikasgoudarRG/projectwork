# Raw Log Sample

## First 20 Lines

```
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906
081109 203519 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.10.6:40524 dest: /10.250.10.6:50010
081109 203519 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.14.224:42420 dest: /10.250.14.224:50010
081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating
081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-1608999687919862906 terminating
081109 203519 145 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6
081109 203519 145 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 91178 from /10.250.19.102
081109 203519 147 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862906 terminating
081109 203519 147 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 91178 from /10.250.14.224
081109 203519 29 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.10.6:50010 is added to blk_-1608999687919862906 size 91178
081109 203519 30 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.111.209:50010 is added to blk_-1608999687919862906 size 91178
081109 203519 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.14.224:50010 is added to blk_-1608999687919862906 size 91178
081109 203520 142 INFO dfs.DataNode$DataXceiver: Receiving block blk_7503483334202473044 src: /10.251.215.16:55695 dest: /10.251.215.16:50010
081109 203520 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_7503483334202473044 src: /10.250.19.102:34232 dest: /10.250.19.102:50010
081109 203520 26 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.split. blk_7503483334202473044
081109 203521 143 INFO dfs.DataNode$DataXceiver: Received block blk_-1608999687919862906 src: /10.251.215.16:52002 dest: /10.251.215.16:50010 of size 91178
081109 203521 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.251.215.16:52002 dest: /10.251.215.16:50010
081109 203521 144 INFO dfs.DataNode$DataXceiver: Receiving block blk_7503483334202473044 src: /10.251.71.16:51590 dest: /10.251.71.16:50010
081109 203521 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_-3544583377289625738 src: /10.250.19.102:39325 dest: /10.250.19.102:50010
```

## Summary

- Total lines: 10000
- Total BlockID occurrences: 10000
- Unique BlockIDs: 1018
- Malformed ratio: 0.0000

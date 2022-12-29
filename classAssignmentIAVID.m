function classLabel=classAssignmentIAVID(label)
%order={'demo','PtBoardScreen','ReadingNotes','UsingLaptop','Walk','WrittingOnBoard'};

switch (label)
    case('interIdle')
    classLabel=1;
    case('PtBoardSc')
    classLabel=2;
    case('PtStudent')
    classLabel=3;
    case('UsingLaptop')
    classLabel=4;
    case('UsingPhone')
    classLabel=5;
    case('Sitting')
    classLabel=6;
    case('Walk')
    classLabel=7;
    case('Writing')
    classLabel=8;              
end

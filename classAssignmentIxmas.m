function classLabel=classAssignment(label)
%check watch, cross arms, scratch head, sit down, get up, turn around, walk, wave, punch, kick, point, pick up and throw

switch (label)
    case('check-watch')
    classLabel=1;
    case('cross-arms')
    classLabel=2;
    case('scratch-head')
    classLabel=3;
    case('sit-down')
    classLabel=4;
    case('get-up')
    classLabel=5;
    case('turn-around')
    classLabel=6;
    case('walk')
    classLabel=7;
    case('wave')
    classLabel=8;
    case('punch')
    classLabel=9;
    case('kick')
    classLabel=10;
%     case('point')
%     classLabel=11;
    case('pick-up')%,,,,',,,,};
    classLabel=11;
    case('throw')
    classLabel=12;                      
end

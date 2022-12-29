function classLabel=classAssignment(label)
%check watch, cross arms, scratch head, sit down, get up, turn around, walk, wave, punch, kick, point, pick up and throw

switch (label)
    case('ClimbLadder')
    classLabel=1;
    case('CrawlOnKnees')
    classLabel=2;
    case('DrawGraffiti')
    classLabel=3;
    case('DrunkWalk')
    classLabel=4;
    case('JumpOverFence')
    classLabel=5;
    case('JumpOverGap')
    classLabel=6;
    case('Kick')
    classLabel=7;
    case('LookInCar')
    classLabel=8;
    case('PickupThrowObject')
    classLabel=9;
    case('PullHeavyObject')
    classLabel=10;
    case('Punch')
    classLabel=11;
    case('RunStop')%,,,,',,,,};
    classLabel=12;
    case('ShotGunCollapse')
    classLabel=13;
    case('SmashObject')
    classLabel=14;
    case('WalkFall')
    classLabel=15;
    case('WalkTurnBack')
    classLabel=16;
    otherwise('WaveArms')
    classLabel=17;                      
end

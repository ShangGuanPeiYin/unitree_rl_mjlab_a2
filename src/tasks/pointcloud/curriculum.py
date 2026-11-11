"""Strict stage budgets: unused iterations are never transferred."""
from dataclasses import dataclass

STAGES = (
    ('P1',0,0.,200,(0.,.2,.3,.4,.5,.6),10),
    ('P2a',4,.10,200,(.2,.3),25), ('P2b',6,.15,200,(.2,.3),25),
    ('P2c',8,.20,200,(.2,.3),25), ('P3a',12,.20,400,(.2,.3,.4),20),
    ('P3b',16,.20,400,(.2,.3,.4),20), ('P3c',20,.20,400,(.2,.3,.4),20),
    ('P4',40,.20,1600,(.2,.3,.4,.5,.6),10),
    ('P5',40,.20,400,(.2,.3,.4,.5,.6),10))

@dataclass
class Curriculum:
    index: int = 0
    iteration: int = 0
    consecutive: int = 0
    stopped: bool = False
    complete: bool = False
    last_validation: int = 0

    @property
    def stage(self):
        return STAGES[self.index]

    def advance_iteration(self):
        if self.iteration % 100 == 0 and self.iteration > self.last_validation:
            raise RuntimeError('Required stage validation has not been recorded')
        if self.stopped or self.complete:
            raise RuntimeError('Curriculum is finished')
        self.iteration += 1
        return self.iteration % 100 == 0

    def validate(self, passed):
        if self.iteration == self.last_validation:
            raise ValueError('Duplicate validation at same iteration')
        if self.iteration == 0 or self.iteration % 100:
            raise ValueError('Validation is required every 100 stage iterations')
        self.last_validation = self.iteration
        self.consecutive = self.consecutive+1 if passed else 0
        if self.consecutive >= 2:
            if self.index == len(STAGES)-1:
                self.complete = True
            else:
                self.index += 1
                self.iteration = self.consecutive = self.last_validation = 0
        elif self.iteration >= self.stage[3]:
            self.stopped = True

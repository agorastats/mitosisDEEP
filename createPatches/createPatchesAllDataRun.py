from createPatches.createPatchesAmida13 import CreatePatchesAmida13
from createPatches.createPatchesBreCaHAD import CreatePatchesBreCaHad
from createPatches.createPatchesGZMH import CreatePatchesGZMH
from createPatches.createPatchesIcpr12 import CreatePatchesIcpr12
from createPatches.createPatchesIcpr14 import CreatePatchesIcpr14
from createPatches.createPatchesMidog21 import CreatePatchesMidog21
from utils.runnable import Main, SequenceRunnable

if __name__ == '__main__':
    Main(
        SequenceRunnable(
   CreatePatchesIcpr12(),
            CreatePatchesIcpr14(),
            CreatePatchesAmida13(),
            CreatePatchesBreCaHad(),
            CreatePatchesMidog21(),
            CreatePatchesGZMH()
        )
    ).run()

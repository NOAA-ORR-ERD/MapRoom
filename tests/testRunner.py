import unittest
import wx

import maproom.library.coordinates
import maproom.library.Projection

alltests = unittest.TestSuite([maproom.library.coordinates.getTestSuite(), maproom.library.Projection.getTestSuite()])

if __name__ == "__main__":
    class TestApp(wx.App):

        def OnInit(self):
            results = unittest.TestResult()
            alltests.run(results)

            numTests = results.testsRun
            numSuccess = numTests - len(results.errors) - len(results.failures)

            print "Successful Tests: %d" % numSuccess
            print "Failed Tests: %d" % (numTests - numSuccess,)

            for failure in results.failures:
                print "------ %s ------" % failure[0]
                print failure[1]

            for error in results.errors:
                print "------ %s ------" % error[0]
                print error[1]

            wx.CallAfter(self.ExitMainLoop)

            return True

    tester = TestApp()
    tester.MainLoop()

package scalation
package modeling
package work

import scala.runtime.ScalaRunTime.stringOf
import scalation.mathstat.*
//import scalation.modeling.Regression
//import scalation.modeling.RidgeRegression
//import scalation.modeling.SymbolicRegression
import scalation.*
import scalation.modeling.ActivationFun.f_aff
//import scalation.modeling.LassoRegression.hp
//import scalation.modeling.RidgeRegression.hp
import scalation.modeling.neuralnet.{Example_Concrete, NeuralNet_2L, NeuralNet_3L, NeuralNet_XL, Optimizer}

//import scala.collection.mutable.Set
//import scala.math.{exp, log, sqrt}
//import SeoulBikeData.*
import scalation.modeling.neuralnet.Example_Concrete.{x_fname}


object airfoilSelfNoise1:
  //define the columns for the airfoil aquatic toxicity
  val airfoil_col = Array ("Frequency","Angle of attack","Chord length","Free-stream velocity","Suction side displacement thickness","Scaled sound pressure level")
  //Load the data into a matrix from the excel
  val airfoil = MatrixD.load("airfoil_self_noise.csv",0)
  //seperate the x and y
  val airfoil_x = airfoil(?, 0 to 4)
  val airfoil_y = airfoil(?, 5)

  val airfoil_mu_x = airfoil_x.mean // column wise mean of x
  val airfoil_mu_y = airfoil_y.mean // mean of y
  val airfoil_x_c = airfoil_x - airfoil_mu_x // centered x (column wise)
  val airfoil_y_c = airfoil_y - airfoil_mu_y // centered y

  val _1 = VectorD.one(airfoil.dim) // vector of all ones
  val airfoil_oxy = _1 +^: airfoil // prepend a column of all ones to xy
  val airfoil_ox = _1 +^: airfoil_x // prepend a column of all ones to x
  val airfoil_yy = MatrixD.fromVector(airfoil_y) // prepend a column of all ones to
  val airfoil_oyy = _1 +^: airfoil_yy

  val airfoil_x_fname: Array [String] = airfoil_col.take (5)
  val airfoil_ox_fname: Array [String] = Array ("intercept") ++ airfoil_x_fname

end airfoilSelfNoise1

import airfoilSelfNoise1._

@main def airfoil_neuralNet_2L2 (): Unit =

  //  println (s"ox = $ox")
  //  println (s"yy = $yy")
  println (s"ox_fname = ${stringOf (airfoil_ox_fname)}")

  for f <- f_aff do                                            // try all activation functions for first layer
    banner (s"airfoil NeuralNet_2L with ${f.name}")
    val mod = NeuralNet_2L.rescale (airfoil_ox, airfoil_yy, airfoil_ox_fname, f = f)  // create model with intercept (else pass x) - rescales
    mod.trainNtest2()()                                      // train and test the model - with auto-tuning

    banner ("airfoil Validation Test")
    println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
  end for

end airfoil_neuralNet_2L2

@main def airfoil_neuralNet_2L1 (): Unit =

  //  println (s"ox = $ox")
  //  println (s"yy = $yy")

  banner ("airfoil NeuralNet_2L")
  //  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
  val mod = NeuralNet_2L.rescale (airfoil_ox, airfoil_yy, airfoil_ox_fname)            // create model with intercept (else pass x) - rescales
  //  mod.trainNtest ()()                                          // train and test the model
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("Cross-Validation")
  FitM.showQofStatTable (mod.crossValidate ())

  println (s"ox_fname = ${stringOf (airfoil_ox_fname)}")

  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${airfoil_ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "smape", "R^2 cv"),
      s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end airfoil_neuralNet_2L1



@main def airfoil_neuralNet_3LTest (): Unit =

  //  println (s"x  = $x")
  //  println (s"yy = $yy")
  println (s"x_fname = ${stringOf (airfoil_x_fname)}")

  //  val mod = new NeuralNet_3L (x, yy, x_fname)                  // create model without intercept
  val mod = NeuralNet_3L.rescale (airfoil_x, airfoil_yy, airfoil_x_fname)              // create model without intercept - rescales

  banner ("NeuralNet_3L: trainNtest")
  mod.trainNtest ()()                                          // train and test the model
  mod.opti.plotLoss ("NeuralNet_3L")                           // loss function vs epochs

  banner ("NeuralNet_3L: trainNtest2")
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  println (mod.summary2 ())                                    // parameter/coefficient statistics
  mod.opti.plotLoss ("NeuralNet_3L")                           // loss function vs epochs

  banner ("NeuralNet_3L: validate")
  println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))

  banner (" NeuralNet_3L: crossValidate")
  val stats = mod.crossValidate ()
  FitM.showQofStatTable (stats)

end airfoil_neuralNet_3LTest

@main def airfoil_neuralNet_3LTest1 (): Unit =

  //  println (s"x  = $x")
  //  println (s"yy = $yy")

  banner ("airfoil NeuralNet_3L")
  Optimizer.hp ("eta") = 0.0025                                // some activation functions need smaller eta
  //val mod = new NeuralNet_3L (airfoil_x, airfoil_yy, airfoil_x_fname)                  // create model without intercept
  val mod = NeuralNet_3L.rescale (airfoil_x, airfoil_yy, airfoil_x_fname)              // create model without intercept - rescales
  banner("airfoil - NeuralNet_3L: trainNtest")
  mod.trainNtest()() // train and test the model
  mod.opti.plotLoss("NeuralNet_3L") // loss function vs epochs

  banner("airfoil - NeuralNet_3L: trainNtest2")
  mod.trainNtest2()() // train and test the model - with auto-tuning
  println(mod.summary2()) // parameter/coefficient statistics
  mod.opti.plotLoss("NeuralNet_3L") // loss function vs epochs

  banner("airfoil - NeuralNet_3L: validate")
  println(FitM.showFitMap(mod.validate()(), QoF.values.map(_.toString)))

  banner("airfoil - NeuralNet_3L: crossValidate")
  val stats = mod.crossValidate()
  FitM.showQofStatTable(stats)

end airfoil_neuralNet_3LTest1

@main def airfoil_neuralNet_3LTest6 (): Unit =
  //  println (s"x  = $x")
  //  println (s"yy = $yy")
  println (s"x_fname = ${stringOf (airfoil_x_fname)}")

  for f <- f_aff do                                            // try all activation functions for first layer
    banner (s"AutoMPG NeuralNet_3L with ${f.name}")
    val mod = NeuralNet_3L.rescale (airfoil_x, airfoil_yy, airfoil_x_fname, f = f)   // create model without intercept - rescales
    mod.trainNtest2 ()()                                     // train and test the model - with auto-tuning

    banner ("AutoMPG Validation Test")
    println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
  end for

end airfoil_neuralNet_3LTest6

@main def airfoil_neuralNet_XLTest2 (): Unit =


  //  println (s"x = $x")
  //  println (s"y = $y")
  println (s"x_fname = ${stringOf (x_fname)}")

  //  val mod = new NeuralNet_XL (x, y, x_fname)                   // create model without intercept)
  val mod = NeuralNet_XL.rescale (airfoil_x, airfoil_yy, airfoil_x_fname)               // create model without intercept - rescales

  banner ("airfoil - NeuralNet_XL: trainNtest")
  mod.trainNtest ()()                                          // train and test the model
  mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs

  banner ("airfoil - NeuralNet_XL: trainNtest2")
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("airfoil - NeuralNet_XL: validate")
  println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))

  banner ("airfoil - NeuralNet_XL: crossValidate")
  val stats = mod.crossValidate ()
  FitM.showQofStatTable (stats)

end airfoil_neuralNet_XLTest2

@main def airfoil_neuralNet_XLTest3 (): Unit =

  import ActivationFun.f_tanh

  val airfoil_ym = airfoil_yy
  Optimizer.hp ("eta") = 0.85                                       // Preceptron and NeuralNet_2L use different optimizers,
  // so different learning rates (eta) are needed.
  banner (s"NeuralNet_2L sigmoid")
  val nn3 = NeuralNet_2L.rescale (airfoil_ox, airfoil_ym)
  nn3.trainNtest ()()                                               // train and test the model

  banner (s"NeuralNet_2L tanh")
  val nn4 = NeuralNet_2L.rescale (airfoil_ox, airfoil_ym, f = f_tanh)
  nn4.trainNtest ()()                                               // train and test the model

  banner (s"NeuralNet_3L sigmoid-id")
  val nn5 = NeuralNet_3L.rescale (airfoil_ox, airfoil_ym)
  nn5.trainNtest ()()                                               // train and test the model

  banner (s"NeuralNet_3L tanh-tanh")
  val nn6 = NeuralNet_3L.rescale (airfoil_ox, airfoil_ym, f = f_tanh, f1 = f_tanh)
  nn6.trainNtest ()()                                               // train and test the model

  banner (s"NeuralNet_XL sigmoid-sigmoid-id")
  val nn7 = NeuralNet_XL.rescale (airfoil_ox, airfoil_ym)
  nn7.trainNtest ()()                                               // train and test the model

  banner (s"NeuralNet_XL tanh-tanh-tanh")
  val nn8 = NeuralNet_XL.rescale (airfoil_ox, airfoil_ym, f = Array (f_tanh, f_tanh, f_tanh))
  nn8.trainNtest ()()                                               // train and test the model

end airfoil_neuralNet_XLTest3


@main def airfoil_RF(): Unit =
  val dmax = 6                                                        // range of depths 1 to dmax
  val qual = new MatrixD (dmax, 3)

  for d <- 1 to dmax do
    banner ("AutoMPG Regression Tree RF with depth d = $d")
    RegressionTree.hp("maxDepth") = d
    RegressionTree.hp("nTrees")   = 7
    val mod = new RegressionTreeRF (airfoil_x, airfoil_y, airfoil_x_fname)                  // create model with intercept (else pass x)
    val qof = mod.trainNtest ()()._2                                // train and test the model
    //      mod.printTree ()                                                // print the regression tree
    //      println (mod.summary ())                                        // parameter/coefficient statistics

    banner (s"AutoMPG Regression Tree RF with d = $d Validation")
    val qof2 = mod.validate ()()                                    // out-of-sampling testing
    val iq = QoF.rSq.ordinal                                        // index for rSq
    qual (d-1) = VectorD (qof(iq), qof(iq+1), qof2(iq))             // R^2, R^2 bar, R^2 os
  end for

  new PlotM (VectorD.range (1, dmax+1), qual.transpose, Array ("R^2", "R^2 bar", "R^2 os"),
    "RegressionTreeRF in-sample, out-of-sample QoF vs. depth", lines = true)
  println (s"RegressionTreeRF: qual = $qual")
end airfoil_RF

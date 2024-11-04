package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf
import scalation.mathstat.*
//import scalation.modeling.Regression
//import scalation.modeling.RidgeRegression
//import scalation.modeling.SymbolicRegression
import scalation.*
import scalation.modeling.ActivationFun.{f_aff, f_id, f_reLU, f_sigmoid, f_tanh}
//import scalation.modeling.LassoRegression.hp
//import scalation.modeling.RidgeRegression.hp
import scalation.modeling.neuralnet.{Example_Concrete, NeuralNet_2L, NeuralNet_3L, NeuralNet_XL, Optimizer}

//import scala.collection.mutable.Set
//import scala.math.{exp, log, sqrt}
import SeoulBikeData.*
import scalation.modeling.neuralnet.Example_Concrete.{x, x_fname}
import scalation.modeling.neuralnet.Optimizer.hp

object Autompg:
  //define the columns for the autompg aquatic toxicity
  //val autompg_col = Array ("Temperature","Humidity","Wind speed","Visibility","Solar Radiation","Rainfall", "Snowfall")

  val autompg_col = Array ("cylinders","displacement","horsepower","weight","acceleration","model year","origin")
  //Load the data into a matrix from the excel
  val autompg = MatrixD.load("auto-mpg.csv",0)
  //seperate the x and y
  val autompg_x = autompg(?, 1 to 7) //18
  val autompg_y = autompg(?, 0)
  val fname = Array ("x")

  val autompg_mu_x = autompg_x.mean // column wise mean of x
  val autompg_mu_y = autompg_y.mean // mean of y
  val autompg_x_c = autompg_x - autompg_mu_x // centered x (column wise)
  val autompg_y_c = autompg_y - autompg_mu_y // centered y

  val _1 = VectorD.one(autompg.dim) // vector of all ones
  val autompg_oxy = _1 +^: autompg // prepend a column of all ones to xy
  val autompg_ox = _1 +^: autompg_x // prepend a column of all ones to x

  val autompg_yy = MatrixD.fromVector(autompg_y) // prepend a column of all ones to
  val autompg_oyy = _1 +^: autompg_yy
  val autompg_x_fname: Array [String] = autompg_col.take(8) //18
  val autompg_ox_fname: Array [String] = Array ("intercept") ++ autompg_x_fname

  val (min_x, max_x) = (autompg.min, autompg.max)
  val autompg_xy_s = scale((min_x, max_x), (0, 1))(autompg) // column-wise scaled to [0.0, 1.0]

  val autompg_xs = autompg(?, 1 to 7) //18
  val autompg_ys = autompg(?, 0)
end Autompg

import Autompg._

@main def autompg_neuralNet_2L_activation (): Unit =

  println (s"ox_fname = ${stringOf (autompg_ox_fname)}")

  for f <- f_aff do                                            // try all activation functions for first layer
    banner (s"Autompg NeuralNet_2L with ${f.name}")
    val mod = NeuralNet_2L.rescale (autompg_ox, autompg_yy, autompg_ox_fname, f = f)  // create model with intercept (else pass x) - rescales
    mod.trainNtest2()()                                      // train and test the model - with auto-tuning

    banner ("Autompg Validation Test")
    println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
  end for

end autompg_neuralNet_2L_activation

@main def autompg_neuralNet_2L_features (): Unit =

  banner ("Autompg NeuralNet_2L")
  val mod = NeuralNet_2L.rescale (autompg_ox, autompg_yy, autompg_ox_fname)            // create model with intercept (else pass x) - rescales
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("Cross-Validation")
  FitM.showQofStatTable (mod.crossValidate ())

  println (s"ox_fname = ${stringOf (autompg_ox_fname)}")

  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${autompg_ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "smape", "R^2 cv"),
      s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end autompg_neuralNet_2L_features

@main def autompg_neuralNet_2L(): Unit =

  println (s"ox_fname = ${stringOf (autompg_ox_fname)}")
  val mod = NeuralNet_2L.rescale (autompg_ox, autompg_yy, autompg_ox_fname)            // create model with intercept (else pass x) - rescales

  banner ("Autompg - NeuralNet_2L: trainNtest")
  mod.trainNtest ()()                                          // train and test the model - manual tuning
  mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs

  banner ("Autompg NeuralNet_2L: trainNtest2")
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs for each eta
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("Autompg - NeuralNet_2L: validate")
  println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))

  banner ("Autompg - NeuralNet_2L: crossValidate")
  val stats = mod.crossValidate ()
  FitM.showQofStatTable (stats)

end autompg_neuralNet_2L


@main def autompg_neuralNet_3LTest (): Unit =

  val ym = MatrixD (autompg_y).transpose
  Optimizer.hp("eta") = 0.01 // set the learning rate (large for small dataset)
  Optimizer.hp("maxEpochs") = 1000
  banner ("Autompg NeuralNet_3L")
  //val mod = new NeuralNet_3L (autompg_x, autompg_yy, autompg_x_fname)                  // create model without intercept
  val mod = NeuralNet_3L.rescale (autompg_x, ym, autompg_x_fname)              // create model without intercept - rescales
  banner("Autompg - NeuralNet_3L: trainNtest")
  mod.trainNtest()() // train and test the model
  mod.opti.plotLoss("NeuralNet_3L") // loss function vs epochs

  banner("Autompg - NeuralNet_3L: trainNtest2")
  mod.trainNtest2()() // train and test the model - with auto-tuning
  println(mod.summary2()) // parameter/coefficient statistics
  mod.opti.plotLoss("NeuralNet_3L") // loss function vs epochs

  banner("Autompg - NeuralNet_3L: validate")
  println(FitM.showFitMap(mod.validate()(), QoF.values.map(_.toString)))

  banner("Autompg - NeuralNet_3L: crossValidate")
  val stats = mod.crossValidate()
  FitM.showQofStatTable(stats)

end autompg_neuralNet_3LTest

@main def autompg_neuralNet_3L_activation (): Unit =
  //  println (s"x  = $x")
  //  println (s"yy = $yy")
  println (s"x_fname = ${stringOf (autompg_x_fname)}")
  val ym = MatrixD (autompg_y).transpose
  Optimizer.hp("eta") = 0.01 // set the learning rate (large for small dataset)
  for f <- f_aff do                                            // try all activation functions for first layer
    banner (s"Autompg NeuralNet_3L with ${f.name}")
    val mod = NeuralNet_3L.rescale (autompg_x, ym, autompg_x_fname, f = f)   // create model without intercept - rescales
    mod.trainNtest2 ()()                                     // train and test the model - with auto-tuning

    banner ("Autompg Validation Test")
    println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
  end for

end autompg_neuralNet_3L_activation

@main def autompg_neuralNet_3L_features (): Unit =
  val ym = MatrixD(autompg_y).transpose
  Optimizer.hp("eta") = 0.01 // set the learning rate (large for small dataset)
  banner ("Autompg NeuralNet_3L")
  val mod = NeuralNet_3L.rescale (autompg_x, ym, autompg_x_fname)              // create model without intercept - rescales
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("Autompg Cross-Validation")
  FitM.showQofStatTable (mod.crossValidate ())

  println (s"x_fname = ${stringOf (x_fname)}")

  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "smape", "R^2 cv"),
      s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end autompg_neuralNet_3L_features
@main def autompg_neuralNet_XLTest (): Unit =

  println (s"x_fname = ${stringOf (x_fname)}")
  val ym = MatrixD (autompg_y).transpose
  //  val mod = new NeuralNet_XL (x, y, x_fname)                   // create model without intercept)
  Optimizer.hp("eta")   = 0.01                                 // set the learning rate (large for small dataset)
  val mod = NeuralNet_XL.rescale (autompg_x, ym, autompg_x_fname, f = Array (f_sigmoid, f_tanh, f_reLU, f_id))              // create model without intercept - rescales

  banner ("Autompg - NeuralNet_XL: trainNtest")
  mod.trainNtest ()()                                          // train and test the model
  mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs

  banner ("Autompg - NeuralNet_XL: trainNtest2")
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("Autompg - NeuralNet_XL: validate")
  println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))

  banner ("Autompg - NeuralNet_XL: crossValidate")
  val stats = mod.crossValidate ()
  FitM.showQofStatTable (stats)

end autompg_neuralNet_XLTest

@main def autompg_neuralNet_XL_activation (): Unit =


  val ym = MatrixD(autompg_y).transpose
  Optimizer.hp("eta")   = 0.01
  Optimizer.hp("bSize") = 20
  println (s"x_fname = ${stringOf (x_fname)}")

  for f <- f_aff; f2 <- f_aff do                               // try all activation functions for first two layers
    banner (s"AutoMPG NeuralNet_XL with ${f.name}")
    val mod = NeuralNet_XL.rescale (autompg_x, ym, autompg_x_fname,
      f = Array (f, f2, f_id))            // create model with intercept (else pass x) - rescales
    mod.trainNtest2 ()()                                     // train and test the model - with auto-tuning

    banner ("AutoMPG Validation Test")
    println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
  end for

end autompg_neuralNet_XL_activation

@main def autompg_neuralNet_XL_features (): Unit =
  val ym = MatrixD(autompg_y).transpose
  Optimizer.hp("eta") = 0.01 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 20
  banner ("AutoMPG NeuralNet_XL")
  //  val mod = new NeuralNet_XL (x, yy, x_fname)                  // create model with intercept (else pass x)
  val mod = NeuralNet_XL.rescale (autompg_x, ym, autompg_x_fname, f= Array(f_sigmoid, f_tanh, f_id) )            // create model with intercept (else pass x) - rescales
  //  mod.trainNtest ()()                                          // train and test the model
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("Cross-Validation")
  FitM.showQofStatTable (mod.crossValidate ())

  println (s"x_fname = ${stringOf (x_fname)}")

  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "smape", "R^2 cv"),
      s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for
end autompg_neuralNet_XL_features

@main def autompg_RF(): Unit =
  val dmax = 6                                                        // range of depths 1 to dmax
  val qual = new MatrixD (dmax, 3)

  for d <- 1 to dmax do
    banner ("AutoMPG Regression Tree RF with depth d = $d")
    RegressionTree.hp("maxDepth") = d
    RegressionTree.hp("nTrees")   = 7
    val mod = new RegressionTreeRF (autompg_x, autompg_y, autompg_x_fname)                  // create model with intercept (else pass x)
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
end autompg_RF


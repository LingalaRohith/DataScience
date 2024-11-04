package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf
import scalation.mathstat.*
//import scalation.modeling.Regression
//import scalation.modeling.RidgeRegression
//import scalation.modeling.SymbolicRegression
import scalation.*
//import scalation.modeling.ActivationFun.{f_aff, f_eLU, f_geLU, f_id, f_lreLU, f_reLU, f_sigmoid, f_softmax, f_tanh}
import scalation.modeling.ActivationFun.{f_aff,f_geLU, f_id,f_reLU, f_sigmoid}
//import scalation.modeling.LassoRegression.hp
//import scalation.modeling.RidgeRegression.hp
import scalation.modeling.neuralnet.{Example_Concrete, NeuralNet_2L, NeuralNet_3L, NeuralNet_XL, Optimizer}

//import scala.collection.mutable.Set
//import scala.math.{exp, log, sqrt}
//import SeoulBikeData.*
import scalation.modeling.neuralnet.Example_Concrete.{x, x_fname}
import scalation.modeling.neuralnet.Optimizer.hp

object SeoulBikeDataNN:
  //define the columns for the bike_data aquatic toxicity
  val bike_data_col = Array("Temperature", "Humidity", "Wind speed", "Visibility", "Solar Radiation", "Rainfall", "Snowfall", "Seasons", "Holiday", "Functioning Day", "hourOfDay", "Rented Bike Count")
  //Load the data into a matrix from the excel
  val bike_data = MatrixD.load("Bike_Data.csv")
  //seperate the x and y
  val bike_data_x = bike_data(?, 1 to 11)
  val bike_data_y = bike_data(?, 0)

  val bike_data_mu_x = bike_data_x.mean // column wise mean of x
  val bike_data_mu_y = bike_data_y.mean // mean of y
  val bike_data_x_c = bike_data_x - bike_data_mu_x // centered x (column wise)
  val bike_data_y_c = bike_data_y - bike_data_mu_y // centered y

  val _1 = VectorD.one(bike_data.dim) // vector of all ones
  val bike_data_oxy = _1 +^: bike_data // prepend a column of all ones to xy
  val bike_data_ox = _1 +^: bike_data_x // prepend a column of all ones to x

  val bike_data_yy = MatrixD.fromVector(bike_data_y) // prepend a column of all ones to
  val bike_data_oyy = _1 +^: bike_data_yy

  val bike_data_x_fname: Array[String] = bike_data_col.take(11)
  val bike_data_ox_fname: Array[String] = Array("intercept") ++ bike_data_x_fname

  val (min_x, max_x) = (bike_data.min, bike_data.max)
  val bike_data_xy_s = scale((min_x, max_x), (0, 1))(bike_data) // column-wise scaled to [0.0, 1.0]

  val bike_data_xs = bike_data(?, 1 to 11) //18
  val bike_data_ys = bike_data(?, 0)

end SeoulBikeDataNN

import SeoulBikeDataNN._

@main def bike_data_neuralNet_2L_activation (): Unit =

  //  println (s"ox = $ox")
  //  println (s"yy = $yy")
  println (s"ox_fname = ${stringOf (bike_data_ox_fname)}")

  for f <- f_aff do                                            // try all activation functions for first layer
    banner (s"Bike Data NeuralNet_2L with ${f.name}")
    val mod = NeuralNet_2L.rescale (bike_data_ox, bike_data_yy, bike_data_ox_fname, f = f)  // create model with intercept (else pass x) - rescales
    mod.trainNtest2()()                                      // train and test the model - with auto-tuning

    banner ("Bike Data Validation Test")
    println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
  end for

end bike_data_neuralNet_2L_activation

@main def bike_data_neuralNet_2L_features (): Unit =
  Optimizer.hp("eta") = 0.001 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 292
  banner ("Bike Data NeuralNet_2L")
  val mod = NeuralNet_2L.rescale (bike_data_ox, bike_data_yy, bike_data_ox_fname)            // create model with intercept (else pass x) - rescales
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("Cross-Validation")
  FitM.showQofStatTable (mod.crossValidate ())

  println (s"ox_fname = ${stringOf (bike_data_ox_fname)}")

  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${bike_data_ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "smape", "R^2 cv"),
      s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end bike_data_neuralNet_2L_features

@main def bike_data_neuralNet_2L(): Unit =

  val ym = MatrixD (bike_data_y).transpose
  Optimizer.hp("eta") = 0.001 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 292
  println (s"ox_fname = ${stringOf (bike_data_ox_fname)}")
  //val mod = new NeuralNet_2L (bike_data_x, bike_data_yy, bike_data_x_fname)                // create model with intercept (else pass x)
  val mod = NeuralNet_2L.rescale (bike_data_ox, ym, bike_data_ox_fname)            // create model with intercept (else pass x) - rescales
  //val mod = NeuralNet_2L.perceptron (bike_data_x, bike_data_y, bike_data_x_fname)          // create model with intercept (else pass x) - rescales

  banner ("BikeData - NeuralNet_2L: trainNtest")
  mod.trainNtest ()()                                          // train and test the model - manual tuning
  mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs

  banner ("BikeData NeuralNet_2L: trainNtest2")
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs for each eta
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("BikeData - NeuralNet_2L: validate")
  println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))

  banner ("BikeData - NeuralNet_2L: crossValidate")
  val stats = mod.crossValidate ()
  FitM.showQofStatTable (stats)

end bike_data_neuralNet_2L



@main def bike_data_neuralNet_3LTest (): Unit =

  val ym = MatrixD (bike_data_y).transpose
  Optimizer.hp("eta") = 0.001 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 292
  Optimizer.hp("maxEpochs") = 1000
  banner ("Bike Data NeuralNet_3L")
  //val mod = new NeuralNet_3L (bike_data_x, bike_data_yy, bike_data_x_fname)                  // create model without intercept
  val mod = NeuralNet_3L.rescale (bike_data_x, ym, bike_data_x_fname)              // create model without intercept - rescales
  banner("Bike Data - NeuralNet_3L: trainNtest")
  mod.trainNtest()() // train and test the model
  mod.opti.plotLoss("NeuralNet_3L") // loss function vs epochs

  banner("Bike Data - NeuralNet_3L: trainNtest2")
  mod.trainNtest2()() // train and test the model - with auto-tuning
  println(mod.summary2()) // parameter/coefficient statistics
  mod.opti.plotLoss("NeuralNet_3L") // loss function vs epochs

  banner("Bike Data - NeuralNet_3L: validate")
  println(FitM.showFitMap(mod.validate()(), QoF.values.map(_.toString)))

  banner("Bike Data - NeuralNet_3L: crossValidate")
  val stats = mod.crossValidate()
  FitM.showQofStatTable(stats)

end bike_data_neuralNet_3LTest

@main def bike_data_neuralNet_3LTest_activation (): Unit =
  //  println (s"x  = $x")
  //  println (s"yy = $yy")
  println (s"x_fname = ${stringOf (bike_data_x_fname)}")
  val ym = MatrixD (bike_data_y).transpose
  Optimizer.hp("eta") = 0.001 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 292

  for f <- f_aff do                                            // try all activation functions for first layer
    banner (s"Bike Data NeuralNet_3L with ${f.name}")
    val mod = NeuralNet_3L.rescale (bike_data_x, ym, bike_data_x_fname, f = f)   // create model without intercept - rescales
    mod.trainNtest2 ()()                                     // train and test the model - with auto-tuning

    banner ("Bike Data Validation Test")
    println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
  end for

end bike_data_neuralNet_3LTest_activation
@main def bike_data_neuralNet_3L_features (): Unit =

  val ym = MatrixD(bike_data_y).transpose
  Optimizer.hp("eta") = 0.001 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 292
  banner ("Bike Data NeuralNet_3L")
  val mod = NeuralNet_3L.rescale (bike_data_x, ym, bike_data_x_fname)              // create model without intercept - rescales
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

end bike_data_neuralNet_3L_features
@main def bike_data_neuralNet_XLTest2 (): Unit =


  println (s"x_fname = ${stringOf (x_fname)}")
  val ym = MatrixD (bike_data_y).transpose
  Optimizer.hp("eta")   = 0.001                                 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 292
  val mod = NeuralNet_XL.rescale (bike_data_xs, ym, bike_data_x_fname, f = Array (f_sigmoid , f_reLU , f_geLU))              // create model without intercept - rescales

  banner ("Bike Data - NeuralNet_XL: trainNtest")
  mod.trainNtest ()()                                          // train and test the model
  mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs

  banner ("Bike Data - NeuralNet_XL: trainNtest2")
  mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
  mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs
  println (mod.summary2 ())                                    // parameter/coefficient statistics

  banner ("Bike Data - NeuralNet_XL: validate")
  println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))

  banner ("Bike Data - NeuralNet_XL: crossValidate")
  val stats = mod.crossValidate ()
  FitM.showQofStatTable (stats)

end bike_data_neuralNet_XLTest2


@main def bike_data_neuralNet_XL_activation (): Unit =


  val ym = MatrixD(bike_data_y).transpose
  Optimizer.hp("eta") = 0.001 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 292
  println (s"x_fname = ${stringOf (x_fname)}")

  for f <- f_aff; f2 <- f_aff do                               // try all activation functions for first two layers
    banner (s"AutoMPG NeuralNet_XL with ${f.name}")
    val mod = NeuralNet_XL.rescale (bike_data_x, ym, bike_data_x_fname,
      f = Array (f, f2, f_id))            // create model with intercept (else pass x) - rescales
    mod.trainNtest2 ()()                                     // train and test the model - with auto-tuning

    banner ("AutoMPG Validation Test")
    println (FitM.showFitMap (mod.validate ()(), QoF.values.map (_.toString)))
  end for

end bike_data_neuralNet_XL_activation

@main def bike_data_neuralNet_XL_features (): Unit =
  val ym = MatrixD(bike_data_y).transpose
  Optimizer.hp("eta") = 0.001 // set the learning rate (large for small dataset)
  Optimizer.hp("bSize") = 292
  banner ("AutoMPG NeuralNet_XL")
  //  val mod = new NeuralNet_XL (x, yy, x_fname)                  // create model with intercept (else pass x)
  val mod = NeuralNet_XL.rescale (bike_data_x, ym, bike_data_x_fname, f = Array(f_sigmoid, f_reLU, f_id))              // create model with intercept (else pass x) - rescales
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

end bike_data_neuralNet_XL_features

@main def SeoulBike_RF(): Unit =
  val dmax = 6                                                        // range of depths 1 to dmax
  val qual = new MatrixD (dmax, 3)

  for d <- 1 to dmax do
    banner ("Seoul Bike Regression Tree RF with depth d = $d")
    RegressionTree.hp("maxDepth") = d
    RegressionTree.hp("nTrees")   = 7
    val mod = new RegressionTreeRF (bike_data_x, bike_data_y, bike_data_x_fname)                  // create model with intercept (else pass x)
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
end SeoulBike_RF

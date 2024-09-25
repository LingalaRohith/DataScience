package scalation
package modeling

//import scalation.modeling.forestAquaticToxicity.{forest, forest_ox, forest_ox_fname, forest_oxy, forest_x, forest_x_fname, forest_y}

//import scala.collection.mutable.Set
//import scala.math.{exp, log}


import scala.runtime.ScalaRunTime.stringOf
import scalation.mathstat.*
//import scalation.modeling.Regression
//import scalation.modeling.RidgeRegression
//import scalation.modeling.SymbolicRegression
import scalation.*
//import scalation.modeling.LassoRegression.hp
//import scalation.modeling.RidgeRegression.hp

import scala.collection.mutable.Set
import scala.math.{exp, log}

object ForestFiresData:
  val forest_col = Array("X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain","month_encoded", "day_encoded","area")
  //Load the data into a matrix from the excel
  val forest = MatrixD.load("output.csv")
  //seperate the x and y
  val forest_x = forest(?, 0 to 13)
  val forest_y = forest(?, 14)
  print(forest_y)

  val forest_mu_x = forest_x.mean // column wise mean of x
  val forest_mu_y = forest_y.mean // mean of y
  val forest_x_c = forest_x - forest_mu_x // centered x (column wise)
  val forest_y_c = forest_y - forest_mu_y // centered y

  val _1 = VectorD.one(forest.dim) // vector of all ones
  val forest_oxy = _1 +^: forest // prepend a column of all ones to xy
  val forest_ox = _1 +^: forest_x // prepend a column of all ones to x

  val forest_x_fname: Array[String] = forest_col.take (14)
  val forest_ox_fname: Array[String] = Array ("intercept") ++ forest_x_fname

end ForestFiresData

import ForestFiresData._

@main def forest_Correlation (): Unit =

  banner ("Variable Names in forest Dataset")
  println (s"xr_fname = ${stringOf (forest_x)}")                     // raw dataset
  println (s"x_fname  = ${stringOf (forest_col)}")                      // origin column removed
  println (s"ox_fname = ${stringOf (forest_ox_fname)}")                     // intercept (1's) added
  banner ("Correlation Analysis: reponse y vs. column x(?, j)")
  for j <- forest_x.indices2 do
    val x_j = forest_x(?, j)
    val correlation = forest_y corr x_j
    val corr2       = correlation * correlation
    println (s"correlation of y vs. x(?, $j) = $correlation \t $corr2 \t ${forest_col(j)}")
    new Plot (x_j, forest_y, null, s"y vs, x(?, $j)")
  end for

end forest_Correlation

@main def forest_NullModel (): Unit =

  banner ("NullModel model: y = b₀")
  val mod = NullModel (forest)                                           // create a null model
  mod.trainNtest ()()                                                // train and test the model

  val forest_yp = mod.predict (forest_x)                                           // predict y for all x rows
  val e  = forest_y - forest_yp                                                    // error/residual vector
  println (s"error e = $e")

  new Plot (null, forest_y, forest_yp, "NullModel")
  new Plot (null, e, null, "NullModel error")

end forest_NullModel

@main def forest_SimpleRegression (): Unit =

  banner ("SimplerRegression model: y = b₁*x₁")
  val mod = SimpleRegression (forest_oxy)                                   // create a SimplerRegression model
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                           // produce summary statistics

end forest_SimpleRegression


@main def forest_Regression (): Unit =

  print(forest_oxy)
  banner ("Regression model: y = b₀ + b₁*x₁ + b₂*x₂ + b₃*x₃ + b₄*x₄ + b₅*x₅ + b₆*x₆")
  val mod = Regression (forest_oxy, forest_ox_fname)()                             // create a Regression Model (with intercept)
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                           // produce summary statistics

end forest_Regression

@main def forest_linear_regression (): Unit =

  val mod = new Regression (forest_ox, forest_y,forest_ox_fname)
  mod.trainNtest ()()
  println(mod.summary ())

  banner ("forest Regression Cross-Validation Test")
  FitM.showQofStatTable (mod.crossValidate ())

  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures (tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${forest_x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Linear Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end forest_linear_regression

@main def forest_ridge_regression (): Unit =

  import RidgeRegression.hp
  println(s"hp = $hp")
  val hp2 = hp.updateReturn("lambda", 1.0)
  println(s"hp2 = $hp2")
  banner("RidgeRegression")
  val mod = new RidgeRegression(forest_x, forest_y, forest_x_fname, hp2)
  mod.trainNtest()() // train and test the model
  println(mod.summary())

  FitM.showQofStatTable (mod.crossValidate ())

  val optLambda2 = mod.findLambda
  println("optimal Lambda value: " + optLambda2) // lambda = 40.96
  hp.updateReturn("lambda", optLambda2._1) // applies first element of optLambda tuple to hyperparam


  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures (tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${forest_x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Ridge Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end forest_ridge_regression

@main def forest_lasso_regression (): Unit =
  banner ("LassoRegression for forest")

  import LassoRegression.hp

  println(s"hp = $hp")
  val hp2 = hp.updateReturn("lambda", 0.64) // try different values
  println(s"hp2 = $hp2")


  val mod = new LassoRegression (forest_x, forest_y,forest_x_fname, hp2)                           // create a Lasso regression model
  mod.trainNtest ()()                                            // train and test the model
  println (mod.summary ())                                       // parameter/coefficient statistics

  banner ("Forward Selection Test")
  val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for LassoRegression", lines = true)
  println (s"rSq = $rSq")
  println (s"best (lambda, sse) = ${mod.findLambda}")

  FitM.showQofStatTable(mod.crossValidate())

  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${forest_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Lasso Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end forest_lasso_regression

@main def forest_SymbolicRegression (): Unit =

  banner("Symbolic Regression")
  //val mod = SymbolicRegression.quadratic (forest_x, forest_y, forest_x_fname)              // add x^2 terms

  val mod = SymbolicRegression(forest_x, forest_y, forest_x_fname,Set(1,1,1,1,1), intercept=true) // add, intercept, cross-terms and given powers
  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics



  FitM.showQofStatTable(mod.crossValidate())
  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${forest_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Symbolic Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end forest_SymbolicRegression

@main def forest_TransformedRegression (): Unit =

  def f(u: Double): Double = -log(1 / u - 1) // transform

  def fi(t: Double): Double = 1 / (1 + exp(-t)) // inverse transform

  val extrem = extreme(forest_y) // (min, max) for y
  val bounds = (0.01, 0.99)
  val yy = scaleV (extrem, bounds)(forest_y)                       // rescale to domain of transform

  banner("Transformed Regression")

  val mod = new TranRegression (forest_ox, yy, forest_ox_fname, Regression.hp, f,fi)

  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics

  val yp2 = mod.predict(forest_ox)
  val e2 = yy - yp2

  val yp2_ = scaleV(bounds, extrem)(yp2)

  val rnk = forest_y.iqsort // rank order for vector y
  val ry = forest_y.reorder(rnk) // actual - red
  val ryp2 = yp2_.reorder(rnk) // TranRegression - blue

  val ys = MatrixD(ry, ryp2)
  new PlotM(null, ys.transpose)

  new Plot(null, e2, null, "e2 vs. t")

  FitM.showQofStatTable(mod.crossValidate())
  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${forest_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Transformed Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end forest_TransformedRegression

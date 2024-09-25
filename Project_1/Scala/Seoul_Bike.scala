package scalation
package modeling

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



object SeoulBikeData:
  //define the columns for the bike_data aquatic toxicity
  val bike_data_col = Array ("Temperature","Humidity","Wind speed","Visibility","Solar Radiation","Rainfall", "Snowfall","Seasons","Holiday","Functioning Day","hourOfDay","Rented Bike Count")
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

  val bike_data_x_fname: Array [String] = bike_data_col.take(11)
  val bike_data_ox_fname: Array [String] = Array ("intercept") ++ bike_data_x_fname

end SeoulBikeData

import SeoulBikeData._

@main def bike_data_Correlation (): Unit =

  banner ("Variable Names in bike_data Dataset")
  println (s"xr_fname = ${stringOf (bike_data_x)}")                     // raw dataset
  println (s"x_fname  = ${stringOf (bike_data_col)}")                      // origin column removed
  println (s"ox_fname = ${stringOf (bike_data_ox_fname)}")                     // intercept (1's) added
  banner ("Correlation Analysis: reponse y vs. column x(?, j)")
  for j <- bike_data_x.indices2 do
    val x_j = bike_data_x(?, j)
    val correlation = bike_data_y corr x_j
    val corr2       = correlation * correlation
    println (s"correlation of y vs. x(?, $j) = $correlation \t $corr2 \t ${bike_data_col(j)}")
    new Plot (x_j, bike_data_y, null, s"y vs, x(?, $j)")
  end for

end bike_data_Correlation

@main def bike_data_NullModel (): Unit =

  banner ("NullModel model: y = b₀")
  val mod = NullModel (bike_data)                                           // create a null model
  mod.trainNtest ()()                                                // train and test the model

  val bike_data_yp = mod.predict (bike_data_x)                                           // predict y for all x rows
  val e  = bike_data_y - bike_data_yp                                                    // error/residual vector
  println (s"error e = $e")

  new Plot (null, bike_data_y, bike_data_yp, "NullModel")
  new Plot (null, e, null, "NullModel error")

end bike_data_NullModel

@main def bike_data_SimpleRegression (): Unit =

  banner ("SimplerRegression model: y = b₁*x₁")
  val mod = SimpleRegression (bike_data_oxy)                                   // create a SimplerRegression model
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                           // produce summary statistics

end bike_data_SimpleRegression


@main def bike_data_Regression (): Unit =

  banner ("Regression model: y = b₀ + b₁*x₁ + b₂*x₂ + b₃*x₃ + b₄*x₄ + b₅*x₅ + b₆*x₆")
  val mod = Regression (bike_data_oxy, bike_data_ox_fname)()                             // create a Regression Model (with intercept)
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                           // produce summary statistics

end bike_data_Regression

@main def bike_data_linear_regression (): Unit =

  val mod = new Regression (bike_data_ox, bike_data_y,bike_data_ox_fname)
  mod.trainNtest ()()
  println(mod.summary ())

  banner ("bike_data Regression Cross-Validation Test")
  FitM.showQofStatTable (mod.crossValidate ())

  for tech <- SelectionTech.values do
    banner (s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures (tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${bike_data_x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end bike_data_linear_regression

@main def bike_data_ridge_regression (): Unit =

  import RidgeRegression.hp
  println(s"hp = $hp")
  val hp2 = hp.updateReturn("lambda", 1.0) // try different values
  println(s"hp2 = $hp2")
  banner("RidgeRegression")
  val mod = new RidgeRegression(bike_data_x, bike_data_y, bike_data_x_fname, hp2)
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
    println (s"k = $k, n = ${bike_data_x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end bike_data_ridge_regression

@main def bike_data_lasso_regression (): Unit =

  import LassoRegression.hp

  println(s"hp = $hp")
  val hp2 = hp.updateReturn("lambda", 0.64) // try different values
  println(s"hp2 = $hp2")


  val mod = new LassoRegression (bike_data_x, bike_data_y,bike_data_x_fname, hp2)                           // create a Lasso regression model
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
    println(s"k = $k, n = ${bike_data_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end bike_data_lasso_regression

@main def bike_data_SymbolicRegression (): Unit =

  banner("Symbolic Regression")
  //val mod = SymbolicRegression.quadratic (bike_data_x, bike_data_y, bike_data_x_fname)              // add x^2 terms

  val mod = SymbolicRegression(bike_data_x, bike_data_y, bike_data_x_fname, Set(1, 1, 1, 1, 1)) // add, intercept, cross-terms and given powers
  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics

  FitM.showQofStatTable(mod.crossValidate())
  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${bike_data_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for symbolic Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end bike_data_SymbolicRegression

@main def bike_data_TransformedRegression (): Unit =

  def f(u: Double): Double = -log(1 / u - 1) // transform

  def fi(t: Double): Double = 1 / (1 + exp(-t)) // inverse transform

  val extrem = extreme(bike_data_y) // (min, max) for y
  val bounds = (0.01, 0.99)
  val yy = scaleV (extrem, bounds)(bike_data_y)                       // rescale to domain of transform

  banner("Transformed Regression")
  //val f = (sqrt _ ,  sq _,    "sqrt")

  val mod = new TranRegression (bike_data_ox, yy, bike_data_ox_fname, Regression.hp, f, fi)

  //val mod = SymbolicRegression(bike_data_x, bike_data_y, bike_data_x_fname, Set(2, 1, 0.5, 2)) // add, intercept, cross-terms and given powers
  mod.trainNtest()() // train and test the model
  println(mod.summary()) // parameter/coefficient statistics

  val yp2 = mod.predict(bike_data_ox)
  val e2 = yy - yp2

  val yp2_ = scaleV(bounds, extrem)(yp2)

  val rnk = bike_data_y.iqsort // rank order for vector y
  val ry = bike_data_y.reorder(rnk) // actual - red
  val ryp2 = yp2_.reorder(rnk) // TranRegression - blue

  val ys = MatrixD(ry, ryp2)
  new PlotM(null, ys.transpose)

  new Plot(null, e2, null, "e2 vs. t")


  banner ("Forward Selection Test")
  val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"), "R^2 vs n for TransformedRegression", lines = true)
  println (s"rSq = $rSq")

  FitM.showQofStatTable(mod.crossValidate())
  for tech <- SelectionTech.values do
    banner(s"Feature Selection Technique Reg: $tech")
    val (cols, rSq) = mod.selectFeatures(tech) // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println(s"k = $k, n = ${bike_data_x.dim2}")
    new PlotM(null, rSq.transpose, Array("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Transformed Regression with $tech", lines = true)
    println(s"$tech: rSq = $rSq")
  end for
end bike_data_TransformedRegression

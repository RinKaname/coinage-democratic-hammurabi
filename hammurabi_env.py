import random
import math

class DemocraticHammurabi:
    """
    Democratic Hammurabi with the Silver Shekel Monetary Economy.
    Decouples monetary capital (silver) from biological calories (grain).
    Rats eat grain, but royal silver vaults are 100% immune!
    """
    def __init__(self, max_years=12):
        self.max_years = max_years
        self.reset()

    def reset(self):
        self.year = 1
        self.population = 300
        self.grain = 6000             # Bushels of food & seed in royal silos
        self.land = 2500              # Acres of farmable land
        self.silver = 30000           # Silver shekels in royal vault
        self.elite_pop = int(self.population * 0.05)
        self.worker_pop = int(self.population * 0.15)
        self.farmer_pop = int(self.population * 0.80)

        self.civilian_grain = 5000    # Bushels in civilian silos

        self.update_demographics_and_prices()
        self.update_grain_price()

        # Dynamic wealth baseline
        self.initial_pop = self.population
        self.initial_wealth = self.silver + (self.land * self.land_price) + int(self.grain * self.grain_price)

        # Faction approvals (0 to 100)
        self.farmers_approval = 50.0
        self.workers_approval = 50.0
        self.elites_approval = 50.0

        self.is_done = False
        self.game_over_reason = ""
        self.starved_total = 0
        self.last_rats_ate = 0
        self.last_harvest_yield = 3
        self.last_immigrants = 0
        self.last_silver = float(self.silver)
        self.last_land_price = float(self.land_price)

        return self._get_state()

    def update_demographics_and_prices(self):
        self.elite_pop = int(self.population * 0.05)
        self.farmer_pop = int(self.population * 0.80)
        self.worker_pop = self.population - self.elite_pop - self.farmer_pop

        self.land_demand = float(self.elite_pop * 20) + float(self.farmer_pop * 10) + float(self.worker_pop * 1)

        # Deterministic land price
        base_land_price = 25.0
        raw_land_price = base_land_price * (self.land_demand / max(1.0, float(self.land)))
        self.land_price = float(max(10.0, min(100.0, raw_land_price)))

    def update_grain_price(self):
        total_supply = float(self.grain + getattr(self, "civilian_grain", 0))
        total_demand = float(self.population * 20)

        # Deterministic Grain Price based on overall supply vs demand
        raw_grain_price = 1.0 * (total_demand / max(1.0, total_supply))
        self.grain_price = float(max(0.5, min(5.0, raw_grain_price)))

    def _get_state(self):
        years_until_election = 4 - (self.year % 4)
        if years_until_election == 4:
            years_until_election = 0

        # 11-Dimensional State Vector
        return [
            float(self.year),
            float(self.population),
            float(self.grain),
            float(self.land),
            float(self.land_price),
            float(self.silver),
            float(self.grain_price),
            float(self.farmers_approval),
            float(self.workers_approval),
            float(self.elites_approval),
            float(years_until_election)
        ]

    def step(self, actions):
        """
        Actions is a list of 4 continuous values:
        - action_land: [-1, 1]. <0 means sell fraction of land for silver, >0 means buy land using silver.
        - action_grain_trade: [-1, 1]. <0 means export/sell grain for silver, >0 means import/buy grain using silver.
        - action_feed: [0, 1]. fraction of silo grain to use for feeding citizens.
        - action_plant: [0, 1]. fraction of remaining silo grain to plant as seed.
        """
        if self.is_done:
            return self._get_state(), 0, self.is_done, {"reason": "Already done"}

        action_land = float(actions[0])
        action_grain_trade = float(actions[1])
        action_feed = float(actions[2])
        action_plant = float(actions[3])

        # Clip actions to expected ranges
        action_land = max(-1.0, min(1.0, action_land))
        action_grain_trade = max(-1.0, min(1.0, action_grain_trade))
        action_feed = max(0.0, min(1.0, action_feed))
        action_plant = max(0.0, min(1.0, action_plant))

        # ----------------------------------------------------------------------
        # 1. Land Market (Traded in Silver Coins)
        # ----------------------------------------------------------------------
        land_changed = 0
        if action_land < 0:
            acres_to_sell = int(abs(action_land) * self.land)
            self.land -= acres_to_sell
            silver_earned = acres_to_sell * self.land_price
            self.silver += silver_earned
            land_changed = -acres_to_sell
        elif action_land > 0:
            max_acres_affordable = int(self.silver // self.land_price)
            acres_to_buy = int(action_land * max_acres_affordable)
            self.land += acres_to_buy
            self.silver -= acres_to_buy * self.land_price
            land_changed = acres_to_buy

        # ----------------------------------------------------------------------
        # 2. Merchant Grain Exchange (Buy/Sell Grain for Silver)
        # ----------------------------------------------------------------------
        grain_traded = 0
        if action_grain_trade < 0:
            bushels_to_sell = int(abs(action_grain_trade) * self.grain)
            self.grain -= bushels_to_sell
            silver_earned = int(bushels_to_sell * self.grain_price)
            self.silver += silver_earned
            grain_traded = -bushels_to_sell
        elif action_grain_trade > 0:
            max_grain_affordable = int(self.silver // self.grain_price)
            bushels_to_buy = int(action_grain_trade * max_grain_affordable)
            self.grain += bushels_to_buy
            self.silver -= int(bushels_to_buy * self.grain_price)
            grain_traded = bushels_to_buy

        # ----------------------------------------------------------------------
        # 3. Feeding Citizens (Workers & Civilians)
        # ----------------------------------------------------------------------
        # Feeding state workers from royal grain
        grain_for_food = int(action_feed * self.grain)
        self.grain -= grain_for_food

        workers_fed = grain_for_food // 20
        workers_starved = max(0, self.worker_pop - workers_fed)

        # Update market grain price before civilian purchases occur
        self.update_grain_price()

        # Civilian food requirements (Farmers + Elites)
        total_civilians = self.elite_pop + self.farmer_pop
        civilian_food_demand = total_civilians * 20

        # Civilians eat available civilian granary reserves first
        grain_consumed = min(self.civilian_grain, civilian_food_demand)
        self.civilian_grain -= grain_consumed
        civilian_deficit = civilian_food_demand - grain_consumed

        civilians_starved = 0
        civilian_grain_bought = 0

        if civilian_deficit > 0:
            # Purchasing power scales down if market price exceeds normal baseline (2.0 silver/bu)
            affordability = min(1.0, 2.0 / self.grain_price)
            affordable_bushels = int(civilian_deficit * affordability)

            # Civilians purchase what they can afford from royal silos
            civilian_grain_bought = min(affordable_bushels, self.grain)
            self.grain -= civilian_grain_bought
            self.silver += int(civilian_grain_bought * self.grain_price)

            # Any remaining shortfall results directly in starvation (20 bu/person)
            unmet_deficit = civilian_deficit - civilian_grain_bought
            civilians_starved = min(total_civilians, unmet_deficit // 20)

        starved = workers_starved + civilians_starved
        self.starved_total = starved

        if starved > 0:
            self.population -= starved
            self.update_demographics_and_prices()

        # Immediate impeachment if starvation exceeds 45%
        if starved > 0.45 * (self.population + starved):
            self.is_done = True
            self.game_over_reason = "Impeached for extreme starvation"
            return self._get_state(), self._calculate_reward(), self.is_done, {"reason": self.game_over_reason}

        # ----------------------------------------------------------------------
        # 4. Planting Seeds (1 bushel per acre, max 10 acres per farmer)
        # ----------------------------------------------------------------------
        grain_for_planting = int(action_plant * self.grain)
        max_workable_by_people = self.farmer_pop * 10
        actual_planted = min(grain_for_planting, self.land, max_workable_by_people)
        self.grain -= actual_planted

        # ----------------------------------------------------------------------
        # 5. Harvest & Rats
        # ----------------------------------------------------------------------
        yield_cycle = [3, 4, 2, 5]
        yield_per_acre = yield_cycle[(self.year - 1) % 4]
        self.last_harvest_yield = yield_per_acre

        harvest = actual_planted * yield_per_acre

        # 40% Tax goes to king, 60% stays with civilian producers
        king_tax = int(harvest * 0.4)
        civilian_harvest = harvest - king_tax

        self.grain += king_tax
        self.civilian_grain += civilian_harvest

        # Rats appear if royal grain silos hold > 5000 bushels
        rats_ate = 0
        if self.grain > 5000:
            rats_ate = int(self.grain * 0.02)
            self.grain -= rats_ate
        self.last_rats_ate = rats_ate

        # ----------------------------------------------------------------------
        # 6. Demographics (Births & Immigrants)
        # ----------------------------------------------------------------------
        immigrants = 0
        if starved == 0:
            wealth_factor = (20 * self.land + self.grain + self.silver) / (100 * self.population + 1)
            immigrants = 5 + int(wealth_factor)
            self.population += immigrants
            self.update_demographics_and_prices()

        self.last_immigrants = immigrants

        # ----------------------------------------------------------------------
        # 7. Faction Approval Updates
        # ----------------------------------------------------------------------
        if land_changed > 0:
            self.farmers_approval += 5
        elif land_changed < 0:
            self.farmers_approval -= 10
        if actual_planted == self.land:
            self.farmers_approval += 5
        if civilians_starved > 0:
            self.farmers_approval -= 20

        if workers_starved > 0:
            self.workers_approval -= (workers_starved / max(1, self.worker_pop)) * 100
        else:
            self.workers_approval += 5

        if self.silver >= self.last_silver:
            self.elites_approval += 3
        else:
            self.elites_approval -= 5

        if self.land_price >= self.last_land_price:
            self.elites_approval += 2
        else:
            self.elites_approval -= 2

        self.last_silver = self.silver
        self.last_land_price = self.land_price

        self.farmers_approval = self._clamp_and_decay_approval(self.farmers_approval)
        self.workers_approval = self._clamp_and_decay_approval(self.workers_approval)
        self.elites_approval = self._clamp_and_decay_approval(self.elites_approval)

        # ----------------------------------------------------------------------
        # 8. Democratic Elections (Every 4 years)
        # ----------------------------------------------------------------------
        if self.year % 4 == 0:
            average_approval = self.get_average_approval()
            if average_approval < 45.0:
                self.is_done = True
                self.game_over_reason = f"Lost election with {average_approval:.1f}% approval"
                return self._get_state(), self._calculate_reward(), self.is_done, {"reason": self.game_over_reason}

        # ----------------------------------------------------------------------
        # 9. Next Year Market Price Fluctuations
        # ----------------------------------------------------------------------
        self.year += 1
        self.update_demographics_and_prices()
        self.update_grain_price()

        if self.year > self.max_years:
            self.is_done = True
            self.game_over_reason = "Completed term successfully!"

        return self._get_state(), self._calculate_reward(), self.is_done, {
            "reason": self.game_over_reason,
            "harvest_yield": yield_per_acre,
            "rats_ate": rats_ate,
            "immigrants": immigrants,
            "workers_starved": workers_starved,
            "civilians_starved": civilians_starved,
            "civilian_grain_bought": civilian_grain_bought,
            "harvest_total": harvest,
            "king_tax": king_tax,
            "civilian_harvest": civilian_harvest,
            "civilian_grain": self.civilian_grain,
        }

    def get_average_approval(self):
        total_voters = max(1, self.farmer_pop + self.worker_pop + self.elite_pop)
        average_approval = ((self.farmer_pop * self.farmers_approval) +
                            (self.worker_pop * self.workers_approval) +
                            (self.elite_pop * self.elites_approval)) / total_voters
        return average_approval

    def _clamp_and_decay_approval(self, approval):
        if approval > 50:
            approval -= (approval - 50) * 0.1
        elif approval < 50:
            approval += (50 - approval) * 0.1
        return max(0.0, min(100.0, approval))

    def _calculate_reward(self):
        survival_bonus = self.year * 100
        wealth_score = (self.silver + (self.land * self.land_price) + (self.grain * self.grain_price)) / 100.0
        pop_score = self.population * 2
        approval_score = min(self.farmers_approval, self.workers_approval, self.elites_approval) * 3.0
        penalty = self.starved_total * 50

        total_score = survival_bonus + wealth_score + pop_score + approval_score - penalty

        if self.is_done and self.year <= self.max_years:
            total_score = total_score / 10.0

        return total_score
